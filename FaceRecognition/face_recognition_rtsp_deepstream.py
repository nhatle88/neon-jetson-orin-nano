#!/usr/bin/env python3
"""
DeepStream Face Recognition Pipeline
---------------------------------------

This script demonstrates a DeepStream-based multi-camera pipeline that:
  • Ingests multiple RTSP streams via GStreamer source bins.
  • Muxes them with nvstreammux.
  • Runs face detection using DeepStream’s nvinfer (with a config file for your face detector).
  • (Optionally) runs tracking.
  • Uses nvdsosd to draw bounding boxes.
  • In a pad probe (before display) the code extracts each detected face, crops the face region,
    and then runs a custom TensorRT-based face recognition inference (using a pre-built engine)
    to compare against a “facebank” (loaded from images).

Make sure that:
  – DeepStream SDK is installed and its Python bindings (pyds) are available.
  – You have a working config file for the face detector (config_infer_primary_face.txt).
  – You have built a TensorRT engine for face recognition saved as “face_recognition.engine”.
  – Your facebank images (JPEGs) are located in the folder defined by FACEBANK_PATH.
"""

import os
import sys
import cv2
import numpy as np
import face_recognition
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GLib, GstVideo
import pyds

# TensorRT and PyCUDA imports for custom TRT inference
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Global constants
FACEBANK_PATH = "facebank"  # Folder containing facebank images (JPEGs)
THRESHOLD = 0.5  # Distance threshold for recognition


def compute_confidence(distance, threshold=THRESHOLD, k=10):
    """
    Computes a confidence percentage from a distance value.
    """
    confidence = 1 / (1 + np.exp(k * (distance - threshold))) * 100
    return confidence


###############################################################################
# FaceRecognition loads the facebank and extracts known embeddings using
# the face_recognition library.
###############################################################################
class FaceRecognition:
    def __init__(self, facebank_path):
        self.facebank = self.load_facebank(facebank_path)

    def load_facebank(self, path):
        facebank = {}
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                # Extract a canonical name from the filename.
                base_name = filename.split("_faceid")[0].lower()
                name_map = {
                    "nhatle": "Nhat Le",
                    "thitran": "Thi Tran",
                    "hungle": "Hubert Le",
                    "stevenle": "Steven Le",
                    "capham": "Ca Pham"
                }
                name = name_map.get(base_name, base_name)
                image_path = os.path.join(path, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    facebank[name] = encodings[0]
                else:
                    print(f"[WARNING] No face found in {filename}")
        return facebank


###############################################################################
# FaceRecognizerTRT loads a pre-built TensorRT engine and runs inference on
# a cropped face image. (Adjust the preprocessing/post‑processing as needed.)
###############################################################################
class FaceRecognizerTRT:
    def __init__(self, engine_path, facebank):
        self.facebank = facebank
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def allocate_buffers(self):
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers.
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def preprocess(self, face_img):
        """
        Preprocess the face image for the recognition model.
        (Example: resize to 112x112, normalize, and transpose to CHW.)
        """
        processed = cv2.resize(face_img, (112, 112))
        processed = processed.astype(np.float32) / 255.0
        processed = np.transpose(processed, (2, 0, 1))  # to CHW
        processed = np.expand_dims(processed, axis=0)  # add batch dimension
        return processed.ravel()  # flatten input

    def infer(self, face_img):
        """
        Runs inference on the preprocessed face image.
        """
        input_data = self.preprocess(face_img)
        np.copyto(self.inputs[0]["host"], input_data)
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        # Run inference.
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back.
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()
        # Assume the output embedding is of length 128.
        embedding = self.outputs[0]["host"]
        return np.array(embedding)

    def recognize(self, face_img):
        """
        Runs the TRT inference on the face image, compares the embedding with the
        facebank, and returns a label (with a confidence percentage).
        """
        embedding = self.infer(face_img)
        label = "Not Registered"
        distances = []
        for name, bank_embedding in self.facebank.items():
            d = np.linalg.norm(embedding - bank_embedding)
            distances.append((name, d))
        if distances:
            best_match, best_distance = min(distances, key=lambda x: x[1])
            if best_distance < THRESHOLD:
                confidence = compute_confidence(best_distance)
                label = f"{best_match} ({round(confidence, 1)}%)"
        return label


###############################################################################
# The pad probe callback runs on each frame buffer that passes through the
# OSD element. It extracts DeepStream metadata (using pyds), loops over detected
# face objects (class_id==0), crops the corresponding region from the frame,
# and then runs face recognition via our TRT engine.
###############################################################################
def osd_sink_pad_buffer_probe(pad, info, user_data):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Retrieve batch metadata from the GstBuffer.
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    # Iterate through frames in the batch.
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Get the frame’s image data as a NumPy array (RGBA).
        frame_image = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # Assume that class_id 0 corresponds to faces.
            if obj_meta.class_id == 0:
                left = int(obj_meta.rect_params.left)
                top = int(obj_meta.rect_params.top)
                width = int(obj_meta.rect_params.width)
                height = int(obj_meta.rect_params.height)
                face_img = frame_image[top:top + height, left:left + width]
                if face_img.size != 0:
                    # Use our TensorRT recognizer to obtain a label.
                    label = user_data.recognizer_trt.recognize(face_img)
                    # Update the metadata text so that OSD displays the label.
                    obj_meta.text_params.display_text = label
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


###############################################################################
# The following helper creates a source bin (for one RTSP stream) that:
#  • Uses rtspsrc → rtph264depay → h264parse → nvv4l2decoder.
#  • Exposes a ghost pad named "src" for linking into nvstreammux.
###############################################################################
def create_source_bin(index, rtsp_uri):
    bin_name = "source-bin-%02d" % index
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        print("Unable to create source bin")
        return None

    # Create the rtspsrc element.
    rtspsrc = Gst.ElementFactory.make("rtspsrc", f"src-{index}")
    rtspsrc.set_property("location", rtsp_uri)
    rtspsrc.set_property("latency", 100)

    # Create additional elements: depay, parse, and decoder.
    depay = Gst.ElementFactory.make("rtph264depay", f"depay-{index}")
    h264parse = Gst.ElementFactory.make("h264parse", f"h264parse-{index}")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", f"decoder-{index}")
    if not depay or not h264parse or not decoder:
        print("Unable to create depay/parse/decoder for stream", index)
        return None

    # Add elements to the bin.
    nbin.add(rtspsrc)
    nbin.add(depay)
    nbin.add(h264parse)
    nbin.add(decoder)

    # Link rtspsrc's dynamic pad to the depay element.
    rtspsrc.connect("pad-added", lambda src, pad: pad.link(depay.get_static_pad("sink")))
    if not depay.link(h264parse):
        print("Failed to link depay to h264parse")
    if not h264parse.link(decoder):
        print("Failed to link h264parse to decoder")

    # Create and add a ghost pad from the decoder's "src" pad.
    ghost_pad = Gst.GhostPad.new("src", decoder.get_static_pad("src"))
    nbin.add_pad(ghost_pad)
    return nbin


###############################################################################
# Main pipeline setup: creates the DeepStream pipeline with multiple RTSP
# sources, nvstreammux, inference (face detection), tracker, OSD, and display.
###############################################################################
def main():
    # Initialize GStreamer.
    Gst.init(None)

    # Create the pipeline.
    pipeline = Gst.Pipeline.new("face-recognition-pipeline")
    if not pipeline:
        print("Unable to create Pipeline")
        return

    # Define RTSP URLs (update with your actual streams/credentials).
    rtsp_urls = [
        "rtsp://admin:password@192.168.1.192:554/cam/realmonitor?channel=1&subtype=1",
        "rtsp://admin:password@192.168.1.185:554/cam/realmonitor?channel=1&subtype=1",
        "rtsp://admin:password@192.168.1.2:554/Streaming/channels/202"
    ]
    num_sources = len(rtsp_urls)
    source_bins = []
    for i, url in enumerate(rtsp_urls):
        source_bin = create_source_bin(i, url)
        if not source_bin:
            print("Source bin creation failed for:", url)
            return
        pipeline.add(source_bin)
        source_bins.append(source_bin)

    # Create nvstreammux to batch the sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("width", 640)
    streammux.set_property("height", 480)
    streammux.set_property("batched-push-timeout", 40000)
    pipeline.add(streammux)

    # Link each source bin’s ghost pad to the streammux sink pads.
    for i, source_bin in enumerate(source_bins):
        src_pad = source_bin.get_static_pad("src")
        if not src_pad:
            print("Unable to get src pad from source bin", i)
        sink_pad = streammux.get_request_pad(f"sink_{i}")
        if not sink_pad:
            print("Unable to get sink pad from streammux")
        src_pad.link(sink_pad)

    # Create nvinfer element for face detection (set your model config file).
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "config_infer_primary_face.txt")
    pipeline.add(pgie)

    # (Optional) Create a tracker element.
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    pipeline.add(tracker)

    # Create the on-screen display element.
    nvdsosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    pipeline.add(nvdsosd)

    # Create a video sink (here, an EGL-based sink for best performance).
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    pipeline.add(sink)

    # Link the elements: streammux → pgie → tracker → nvdsosd → sink.
    if not streammux.link(pgie):
        print("Streammux and pgie could not be linked.")
        return
    if not pgie.link(tracker):
        print("Pgie and tracker could not be linked.")
        return
    if not tracker.link(nvdsosd):
        print("Tracker and nvdsosd could not be linked.")
        return
    if not nvdsosd.link(sink):
        print("Nvdsosd and sink could not be linked.")
        return

    # Create our FaceRecognition and FaceRecognizerTRT objects.
    face_recog = FaceRecognition(FACEBANK_PATH)
    recognizer_trt = FaceRecognizerTRT("face_recognition.engine", face_recog.facebank)

    # Create an object to pass to the probe callback.
    class ProbeData:
        pass

    probe_data = ProbeData()
    probe_data.recognizer_trt = recognizer_trt

    # Add a probe on the sink pad of the OSD element.
    osd_sink_pad = nvdsosd.get_static_pad("sink")
    if not osd_sink_pad:
        print("Unable to get sink pad from nvdsosd")
    else:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, probe_data)

    # Start playing the pipeline.
    pipeline.set_state(Gst.State.PLAYING)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except Exception as e:
        print("Error: ", e)
    # Clean up on exit.
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    main()