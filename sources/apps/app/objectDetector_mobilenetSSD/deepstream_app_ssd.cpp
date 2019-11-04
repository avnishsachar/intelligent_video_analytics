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

#define TRACKER_CONFIG_FILE "config_tracker_ssd.txt"
#define MAX_TRACKING_ID_LEN 16


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

#define USER_ARRAY_SIZE 16

/** set the user metadata type */
#define NVDS_USER_FRAME_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
guint frame_number = 0;

const gchar pgie_classes_str[PGIE_DETECTED_CLASS_NUM][32] =
    { "unlabeled", "person" };

char* rtsp_links[] = { "NULL", "rtsp://admin:admin123@192.168.0.103", "rtsp://admin:admin123@192.168.0.103/Streaming/Channels/101", "rtsp://admin:admin123@192.168.0.101/Streaming/Channels/101", NULL };

// #define FPS_PRINT_INTERVAL 300

/* Function to find distance between bbox centroid and lines */
float shortest_distance(float x1, float y1, 
                       float a, float b,  
                       float c) 
{ 
    float d = (a * x1 + b * y1 + c) /  
             (sqrt(a * a + b * b)); 
    return d;
}

static gpointer copy_user_meta(gpointer data, gpointer user_data);
static gpointer release_user_meta(gpointer data, gpointer user_data);

/* copy function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer copy_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
  gchar *src_user_metadata = (gchar*)user_meta->user_meta_data;
  gchar *dst_user_metadata = (gchar*)g_malloc0(USER_ARRAY_SIZE);
  memcpy(dst_user_metadata, src_user_metadata, USER_ARRAY_SIZE);
  return (gpointer)dst_user_metadata;
}

/* release function set by user. "data" holds a pointer to NvDsUserMeta*/
static gpointer release_user_meta(gpointer data, gpointer user_data)
{
  NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
  if(user_meta->user_meta_data) {
    g_free(user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
  }
}
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
  guint source_id;
  NvDsDisplayMeta *display_meta = NULL;
  gchar *msg = NULL;
  NvDsMetaList * l_user_meta = NULL;
  NvDsUserMeta *user_meta = NULL;
  gchar *user_meta_data = NULL;
  int i = 0;
  int j = 0;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    // /* Iterate user metadata in frames to search PGIE's tensor metadata */
    // for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
    //     l_user != NULL; l_user = l_user->next) {
    //   NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
    //   if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
    //     continue;

    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
        person_count++;
        num_rects++;
      }
    }
    NvOSD_LineParams *line_params  = display_meta->line_params;
    for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL;
        l_user_meta = l_user_meta->next) {
      user_meta = (NvDsUserMeta *) (l_user_meta->data);
      user_meta_data = (gchar *)user_meta->user_meta_data;

      if(user_meta->base_meta.meta_type == NVDS_USER_FRAME_META_EXAMPLE)
      {
        // g_print("\n************ Retrieving user_meta_data array of 16 on osd sink pad\n");
        // for(i = 0; i < USER_ARRAY_SIZE; i++) {
        //   g_print("user_meta_data [%d] = %d\n", i, user_meta_data[i]);
        
        // }
        // g_print("\n");
        line_params[0].x1 = -160;
        line_params[0].y1 = 1076;
        line_params[0].x2 = 100;
        line_params[0].y2 = 100;
        line_params[0].line_width = user_meta_data[4];
        line_params[0].line_color = (NvOSD_ColorParams){user_meta_data[5], user_meta_data[6], user_meta_data[7], user_meta_data[8]};
        display_meta->num_lines++;
      }
    }
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    int offset = 0;
    source_id = frame_meta->source_id;

    txt_params->display_text = (gchar *) g_malloc0 (MAX_DISPLAY_LEN);
    offset =
        snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "Persons: %d",
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
  // g_print ("Frame Number = %d"
  //     "Person Count = %d\n",
  //     frame_number, person_count);
  // g_object_get (G_OBJECT (u_data), "last-message", &msg, NULL);
  // if (msg != NULL) {
  //   g_print ("Fps info: %s\n", msg);
  // }
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
  guint bbox_x, bbox_y;
  gfloat bbox_d;
  NvDsUserMeta *user_meta = NULL;
  NvDsMetaType user_meta_type = NVDS_USER_FRAME_META_EXAMPLE;
  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    /* Acquire NvDsUserMeta user meta from pool */
    user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    /* Set NvDsUserMeta below */
    // user_meta->user_meta_data = (void *)set_metadata_ptr();
    // user_meta->base_meta.meta_type = user_meta_type;
    // user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)copy_user_meta;
    // user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;

    // NvDecoderMeta *line_meta = (NvDecoderMeta *)g_malloc0(sizeof(NvDecoderMeta));
    gchar *line_meta = (gchar*)g_malloc0(USER_ARRAY_SIZE);
    /* Add dummy metadata */
    line_meta[0] = 50 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH; //x1
    line_meta[1] = 40 * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT; //y1
    line_meta[2] = 250 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH; //x2
    line_meta[3] = 255 * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT; //y2
    line_meta[4] = 4; //width
    line_meta[5] = 0.0; //red
    line_meta[6] = 0.0; //green
    line_meta[7] = 1.0; //blue
    line_meta[8] = 1.0; //transparency

    /* Set NvDsUserMeta below */
    user_meta->user_meta_data = (gpointer *)line_meta;
    user_meta->base_meta.meta_type = user_meta_type;
    user_meta->base_meta.copy_func = (NvDsMetaCopyFunc)copy_user_meta;
    user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;

    /* We want to add NvDsUserMeta to frame level */
    nvds_add_user_meta_to_frame(frame_meta, user_meta); 

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

          /* Extracting bbox centroid coordinates */
          bbox_x = rect_params.left - (rect_params.width/2);
          bbox_y = rect_params.top - rect_params.height;
          // g_print("X coordinate: %f Y coordinate: %f\n", bbox_x, bbox_y);
          // g_print("left:%d, top:%d, width:%d, height:%d\n\n",rect_params.left, rect_params.top, rect_params.width, rect_params.height);
          
          /*Critical Line */
          line_meta[0] = 50 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH; //x1
          line_meta[1] = 40 * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT; //y1
          line_meta[2] = 250 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH; //x2
          line_meta[3] = 255 * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT; //y2
          
          gfloat x = 3; 
          gfloat y = 4; 
          gfloat A = 4; 
          gfloat B = 3; 
          gfloat C = 4;
          bbox_d = shortest_distance(bbox_x, bbox_y, A, B, C); 
          // g_print("DISTANCE d: %f\n", bbox_d);

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

/* Tracker config parsing */
#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }

#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
  gchar abs_cfg_path[PATH_MAX + 1];
  gchar *abs_file_path;
  gchar *delim;

  if (file_path && file_path[0] == '/') {
    return file_path;
  }

  if (!realpath (cfg_file_path, abs_cfg_path)) {
    g_free (file_path);
    return NULL;
  }

  // Return absolute path of config file if file_path is NULL.
  if (!file_path) {
    abs_file_path = g_strdup (abs_cfg_path);
    return abs_file_path;
  }

  delim = g_strrstr (abs_cfg_path, "/");
  *(delim + 1) = '\0';

  abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
  g_free (file_path);

  return abs_file_path;
}

static gboolean
set_tracker_properties (GstElement *nvtracker)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  gchar **keys = NULL;
  gchar **key = NULL;
  GKeyFile *key_file = g_key_file_new ();

  if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
          &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    return FALSE;
  }

  keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
  CHECK_ERROR (error);

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
      gint width =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_WIDTH, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
      gint height =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_HEIGHT, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
      char* ll_config_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
      char* ll_lib_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                g_key_file_get_string (key_file,
                    CONFIG_GROUP_TRACKER,
                    CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
      gboolean enable_batch_process =
          g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
          CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
      CHECK_ERROR (error);
      g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                    enable_batch_process, NULL);
    } else {
      g_printerr ("Unknown key '%s' for group [%s]", *key,
          CONFIG_GROUP_TRACKER);
    }
  }

  ret = TRUE;
done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  if (!ret) {
    g_printerr ("%s failed", __func__);
  }
  return ret;
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
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    default:
      break;
  }
  return TRUE;
}
static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}
static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
  if (g_strstr_len (name, -1, "nvv4l2decoder") == name) {
    g_print ("Seting bufapi_version\n");
    g_object_set (object, "bufapi-version", TRUE, NULL);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =
      NULL, *decoder = NULL, *streammux = NULL, *nvsink = NULL, *sink = NULL, *pgie =
      NULL, *nvvidconv = NULL, *nvosd = NULL, *tiler = NULL, *nvtracker = NULL;
  g_print ("With tracker\n");
#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL, *tiler_sink_pad = NULL;
  guint i;
  guint pgie_batch_size;
  guint tiler_rows;
  guint tiler_columns;

  /* Check input arguments */
  if (argc < 1) {
    g_printerr ("Usage: %s\n", argv[0]);
    return -1;
  }

  argv = rtsp_links;
  argc = sizeof(rtsp_links)/sizeof(rtsp_links[0]) - 1;
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
  gst_bin_add (GST_BIN (pipeline), streammux);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };
    GstElement *source_bin = create_source_bin (i, argv[i + 1]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* We need to have a tracker to track the identified objects */
  nvtracker = gst_element_factory_make ("nvtracker", "tracker");
  
  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  transform = gst_element_factory_make ("queue", "queue");
#endif
  nvsink = gst_element_factory_make ("nvoverlaysink", "nvvideo-renderer");
  // nvsink = gst_element_factory_make ("fakesink", "nvvideo-renderer");
  sink = gst_element_factory_make ("fpsdisplaysink", "fps-display");

  if (!pgie || !nvtracker || !nvvidconv || !nvosd || !sink ||
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
      MUXER_OUTPUT_HEIGHT, "batch-size", num_sources, "live-source", TRUE,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Set all the necessary properties of the nvinfer element,
   * Output tensor meta can be enabled by set "output-tensor-meta=true" here
   * or enable this attribute in config file. with that we can probe PGIE and
   * SGIEs buffer to parse tensor output data of models */
  g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE,
      "output-tensor-meta", TRUE, NULL);

  /* Set necessary properties of the tracker element. */
  if (!set_tracker_properties(nvtracker)) {
    g_printerr ("Failed to set tracker properties. Exiting.\n");
    return -1;
  }

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  if (num_sources %2 == 0) {
    guint tiler_rows = sqrt (num_sources);
  }
  else {
    tiler_rows = sqrt (num_sources) + 1.0;
  }
  tiler_columns = ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (sink), "text-overlay", FALSE, "video-sink", nvsink, "sync", FALSE, NULL);
      
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  /* decoder | pgie1 | sgie1 | sgie2 | sgie3 | etc.. */
#ifdef PLATFORM_TEGRA
  gst_bin_add_many (GST_BIN (pipeline), pgie, nvtracker, tiler, nvvidconv, nvosd, transform, sink,
      NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, pgie, nvtracker, tiler, nvvidconv, nvosd, transform, sink,
          NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
gst_bin_add_many (GST_BIN (pipeline), pgie, nvtracker, tiler, nvvidconv, nvosd, sink,
      NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, pgie, nvtracker, tiler, nvvidconv, nvosd, sink,
          NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#endif

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
  g_print ("Now playing:");
  for (i = 0; i < num_sources; i++) {
    g_print (" %s,", argv[i + 1]);
  }
  g_print ("\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
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