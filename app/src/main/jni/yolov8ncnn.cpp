// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

extern "C" {
// FIXME DeleteGlobalRef is missing for objCls
static Yolo* g_yolo = 0;
static ncnn::Mutex lock;
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;
static const char* class_names[] = {"shi", "hun", "other"};
static int img_input_size = 416;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");
    ncnn::create_gpu_instance();
    if (!g_yolo)
         g_yolo = new Yolo;
    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(lock);
        ncnn::destroy_gpu_instance();
        delete g_yolo;
        g_yolo = 0;
    }
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager)
{

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);
    const float mean_vals[3] ={103.53f, 116.28f, 123.675f};
    const float norm_vals[3] ={ 1 / 255.f, 1 / 255.f, 1 / 255.f };
    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (!g_yolo)
           g_yolo = new Yolo;
        g_yolo->load(mgr, img_input_size, mean_vals, norm_vals);
    }
    // init jni glue
        jclass localObjCls = env->FindClass("com/megvii/yoloXncnn/YOLOXncnn$Obj");
        objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

        constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/megvii/yoloXncnn/YOLOXncnn;)V");

        xId = env->GetFieldID(objCls, "x", "F");
        yId = env->GetFieldID(objCls, "y", "F");
        wId = env->GetFieldID(objCls, "w", "F");
        hId = env->GetFieldID(objCls, "h", "F");
        labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
        probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}


JNIEXPORT jobjectArray JNICALL Java_com_tencent_yolov8ncnn_Yolov8Ncnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap,jboolean use_gpu,float prob_threshold, float nms_threshold){
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

   if (g_yolo)
    {
       double start_time = ncnn::get_current_time();
       std::vector<Object> objects;
       g_yolo->detect(bitmap, objects, use_gpu, prob_threshold,nms_threshold);
       jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

       for (size_t i=0; i<objects.size(); i++)
       {
          jobject jObj = env->NewObject(objCls, constructortorId, thiz);

          env->SetFloatField(jObj, xId, objects[i].rect.x);
          env->SetFloatField(jObj, yId, objects[i].rect.y);
          env->SetFloatField(jObj, wId, objects[i].rect.width);
          env->SetFloatField(jObj, hId, .rect.height);
          env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
          env->SetFloatField(jObj, probId, objects[i].prob);

          env->SetObjectArrayElement(jObjArray, i, jObj);

          double elasped = ncnn::get_current_time() - start_time;
          __android_log_print(ANDROID_LOG_DEBUG, "YOLOXncnn", "%.2fms   detect", elasped);
    }else{
          __android_log_print(ANDROID_LOG_DEBUG, "Yolov8Ncnn", "Detect load model failed");
    }
    return jObjArray;
   }
}


}
