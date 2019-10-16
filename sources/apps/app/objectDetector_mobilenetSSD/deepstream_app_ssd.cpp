/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>

#include <math.h>

#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"

#include <opencv2/objdetect/objdetect.hpp>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"

#define PGIE_CONFIG_FILE  "config_infer_primary_ssd.txt"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_PERSON 1

#define PGIE_DETECTED_CLASS_NUM 2

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720


#define PGIE_NET_WIDTH 640
#define PGIE_NET_HEIGHT 368

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 4000000

gint frame_number = 0;
const gchar pgie_classes_str[PGIE_DETECTED_CLASS_NUM][32] =
    { "unlabeled", "person" };


/* This is the buffer probe function that we have registered on the sink pad
 * of the OSD element. All the infer elements in the pipeline shall attach
 * their metadata to the GstBuffer, here we will iterate & process the metadata
 * forex: class ids to strings, counting of class_id objects etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
        person_count++;
        num_rects++;
      }
    }
    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (gchar *) g_malloc0 (MAX_DISPLAY_LEN);
    offset =
        snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ",
        person_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    txt_params->font_params.font_name = (gchar *) "Serif";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame (frame_meta, display_meta);
  }


  g_print ("Frame Number = %d Number of objects = %d "
      "Person Count = %d\n",
      frame_number, num_rects, person_count);
  frame_number++;
  return GST_PAD_PROBE_OK;
}

extern "C"
    bool NvDsInferParseCustomSSD (std::vector < NvDsInferLayerInfo >
    const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector < NvDsInferObjectDetectionInfo > &objectList);

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. PGIE element in the pipeline shall attach
 * its NvDsInferTensorMeta to each frame metadata on GstBuffer, here we will
 * iterate & parse the tensor data to get detection bounding boxes. The result
 * would be attached as object-meta(NvDsObjectMeta) into the same frame metadata.
 */
static GstPadProbeReturn
pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  static guint use_device_mem = 0;
  static NvDsInferNetworkInfo networkInfo
  {
  PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
  static NvDsInferParseDetectionParams detectionParams
  {
    2,
    {
  0.2, 0.2}};
  static float groupThreshold = 1;
  static float groupEps = 0.2;

  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    /* Iterate user metadata in frames to search PGIE's tensor metadata */
    for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
        l_user != NULL; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;

      /* convert to tensor metadata */
      NvDsInferTensorMeta *meta =
          (NvDsInferTensorMeta *) user_meta->user_meta_data;
      for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];
        if (use_device_mem) {
          cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
              info->dims.numElements * 4, cudaMemcpyDeviceToHost);
        }
      }
      /* Parse output tensor and fill detection results into objectList. */
      std::vector < NvDsInferLayerInfo >
          outputLayersInfo (meta->output_layers_info,
          meta->output_layers_info + meta->num_output_layers);
      std::vector < NvDsInferObjectDetectionInfo > objectList;
      NvDsInferParseCustomSSD (outputLayersInfo, networkInfo,
          detectionParams, objectList);

      /* Seperate detection rectangles per class for grouping. */
      std::vector < std::vector <
          cv::Rect >> objectListClasses (PGIE_DETECTED_CLASS_NUM);
    for (auto & obj:objectList) {
        objectListClasses[obj.classId].emplace_back (obj.left, obj.top,
            obj.width, obj.height);
      }

      for (uint32_t c = 0; c < objectListClasses.size (); ++c) {
        auto & objlist = objectListClasses[c];
        if (objlist.empty ())
          continue;

        /* Merge and cluster similar detection results */
        cv::groupRectangles (objlist, groupThreshold, groupEps);

        /* Iterate final rectangules and attach result into frame's obj_meta_list. */
      for (const auto & rect:objlist) {
          NvDsObjectMeta *obj_meta =
              nvds_acquire_obj_meta_from_pool (batch_meta);
          obj_meta->unique_component_id = meta->unique_id;
          obj_meta->confidence = 0.0;

          /* This is an untracked object. Set tracking_id to -1. */
          obj_meta->object_id = UNTRACKED_OBJECT_ID;
          obj_meta->class_id = c;

          NvOSD_RectParams & rect_params = obj_meta->rect_params;
          NvOSD_TextParams & text_params = obj_meta->text_params;

          /* Assign bounding box coordinates. */
          rect_params.left = rect.x * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
          rect_params.top = rect.y * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
          rect_params.width = rect.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
          rect_params.height =
              rect.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;

          /* Border of width 3. */
          rect_params.border_width = 3;
          rect_params.has_bg_color = 0;
          rect_params.border_color = (NvOSD_ColorParams) {
          1, 0, 0, 1};

          /* display_text requires heap allocated memory. */
          text_params.display_text = g_strdup (pgie_classes_str[c]);
          /* Display text above the left top corner of the object. */
          text_params.x_offset = rect_params.left;
          text_params.y_offset = rect_params.top - 10;
          /* Set black background for the text. */
          text_params.set_bg_clr = 1;
          text_params.text_bg_clr = (NvOSD_ColorParams) {
          0, 0, 0, 1};
          /* Font face, size and color. */
          text_params.font_params.font_name = (gchar *) "Serif";
          text_params.font_params.font_size = 11;
          text_params.font_params.font_color = (NvOSD_ColorParams) {
          1, 1, 1, 1};
          nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
        }
      }
    }
  }
  use_device_mem = 1 - use_device_mem;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =
      NULL, *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie =
      NULL, *nvvidconv = NULL, *nvosd = NULL, *tiler = NULL;
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL, *tiler_sink_pad = NULL;
  guint i;

  /* Check input arguments */
  if (argc < 2) {
    g_printerr
        ("Usage: %s <elementary H264 file 1> ... <elementary H264 file n>\n",
        argv[0]);
    return -1;
  }

  guint num_sources = argc - 1;

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will be a container of other elements */
  pipeline = gst_pipeline_new ("mobilenet_ssd-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "tiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");

  if (!pgie || !nvvidconv || !nvosd || !sink ||
      !tiler) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
#ifdef PLATFORM_TEGRA
  if (!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif


  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * Output tensor meta can be enabled by set "output-tensor-meta=true" here
   * or enable this attribute in config file. with that we can probe PGIE and
   * SGIEs buffer to parse tensor output data of models */
  g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE,
      "output-tensor-meta", TRUE, "batch-size", num_sources, NULL);

  guint rows = sqrt (num_sources);
  g_object_set (G_OBJECT (tiler), "rows", rows, "columns",
      (guint) ceil (1.0 * num_sources / rows), "width", 1920, "height", 1080,
      NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  /* decoder | pgie1 | sgie1 | sgie2 | sgie3 | etc.. */
#ifdef PLATFORM_TEGRA
  gst_bin_add (GST_BIN (pipeline), transform);
#endif
  gst_bin_add_many (GST_BIN (pipeline),
      streammux, pgie, tiler, nvvidconv, nvosd, sink, NULL);

  for (i = 0; i < num_sources; i++) {
    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", NULL);

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    h264parser = gst_element_factory_make ("h264parse", NULL);

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", NULL);
    gst_bin_add_many (GST_BIN (pipeline), source, h264parser, decoder, NULL);

    if (!source || !h264parser || !decoder) {
      g_printerr ("One element could not be created. Exiting.\n");
      return -1;
    }

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16];
    sprintf (pad_name_sink, "sink_%d", i);
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
      g_printerr ("Decoder request src pad failed. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    /* Link the elements together */
    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
      g_printerr ("Elements could not be linked: 1. Exiting.\n");
      return -1;
    }
    /* Set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", argv[i + 1], NULL);
  }

  if (!gst_element_link_many (streammux, pgie, tiler, nvvidconv, nvosd,
#ifdef PLATFORM_TEGRA
          transform,
#endif
          sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }

  /* Add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the sink pad of tiler element which is just after all SGIE elements.
   * Since by that time, GstBuffer would have had got all SGIEs tensor
   * metadata. */
  tiler_sink_pad = gst_element_get_static_pad (tiler, "sink");
  gst_pad_add_probe (tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      pgie_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing: %s\n", argv[1]);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Iterate */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}