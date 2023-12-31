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

package com.tencent.yolov8ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.media.ExifInterface;
import android.media.Image;
import android.net.Uri;
import android.os.Bundle;
import android.os.Message;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;


import com.tencent.yolov8ncnn.yolov8.DectectResult;
import com.tencent.yolov8ncnn.yolov8.OverStatusData;
import com.tencent.yolov8ncnn.yolov8.Result;
import com.tencent.yolov8ncnn.yolov8.SupportOnnx;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends Activity
{
    private static final int SELECT_IMAGE = 1;
    private ImageView imageView;
    private Bitmap bitmap = null;
    private Bitmap yourSelectedImage = null;
    private SupportOnnx supportOnnx;
    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private TextView textView,overtv;


    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        load_model();

        imageView = (ImageView) findViewById(R.id.imageView);
        textView = findViewById(R.id.itv);
        overtv = findViewById(R.id.overtv);

        Button buttonImage = (Button) findViewById(R.id.buttonImage);
        buttonImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });

        Button buttonDetect = (Button) findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null)
                    return;

                ArrayList<Result> objects = YoloDectect(yourSelectedImage);

                showObjects(objects);
            }
        });


    }


    private void showObjects(ArrayList<Result> objects)
    {
        if (objects == null)
        {
            imageView.setImageBitmap(bitmap);
            return;
        }

        // draw objects on bitmap
        Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        final int[] colors = new int[] {
                Color.rgb( 54,  67, 244),
                Color.rgb(176,  39, 156),
                Color.rgb(243, 150,  33),
                Color.rgb(244, 169,   3),
                Color.rgb(212, 188,   0),
                Color.rgb(136, 150,   0),
                Color.rgb( 80, 175,  76),
                Color.rgb( 74, 195, 139),
                Color.rgb( 57, 220, 205),
                Color.rgb( 59, 235, 255),
                Color.rgb(  7, 193, 255),
                Color.rgb(  0, 152, 255),
                Color.rgb( 34,  87, 255),
                Color.rgb( 72,  85, 121),
                Color.rgb(158, 158, 158),
                Color.rgb(139, 125,  96)
        };

        Canvas canvas = new Canvas(rgba);

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4);

        Paint textbgpaint = new Paint();
        textbgpaint.setColor(Color.WHITE);
        textbgpaint.setStyle(Paint.Style.FILL);

        Paint textpaint = new Paint();
        textpaint.setColor(Color.BLACK);
        textpaint.setTextSize(26);
        textpaint.setTextAlign(Paint.Align.LEFT);

        for (int i = 0; i < objects.size(); i++)
        {
            paint.setColor(colors[i % 3]);

            canvas.drawRect(objects.get(i).getRectF().left, objects.get(i).getRectF().top, objects.get(i).getRectF().right, objects.get(i).getRectF().bottom, paint);

            // draw filled text inside image
            {
                String text = objects.get(i).getLabel() + " = " + String.format("%.1f", objects.get(i).getScore() * 100) + "%";

                float text_width = textpaint.measureText(text);
                float text_height = - textpaint.ascent() + textpaint.descent();

                float x = objects.get(i).getRectF().left;
                float y = objects.get(i).getRectF().top - text_height;
                if (y < 0)
                    y = 0;
                if (x + text_width > rgba.getWidth())
                    x = rgba.getWidth() - text_width;

                canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint);

                canvas.drawText(text, x, y - textpaint.ascent(), textpaint);
            }
        }

        imageView.setImageBitmap(rgba);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            try
            {
                if (requestCode == SELECT_IMAGE) {
                    bitmap = decodeUri(selectedImage);

                    yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                    imageView.setImageBitmap(bitmap);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 416;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);

        // Rotate according to EXIF
        int rotate = 0;
        try
        {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(selectedImage));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }
        }
        catch (IOException e)
        {
            Log.e("MainActivity", "ExifInterface IOException");
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }


    @Override
    public void onPause()
    {
        super.onPause();


    }
    @Override
    protected void onStop() {

        try {
            ortSession.endProfiling();
        } catch (OrtException e) {
            e.printStackTrace();
        }
        super.onStop();
    }
    @Override
    public void onDestroy() {

        try {
            ortSession.close();
            ortEnvironment.close();
        } catch (OrtException e) {
            e.printStackTrace();
        }
        super.onDestroy();
    }

    public ArrayList<Result> YoloDectect(Bitmap image) {
        if (image != null) {
            OverStatusData overStatusData= OverStatusData.getInstance();
            overStatusData.setCenterx(image.getWidth()/2);
//            Bitmap bitmap = supportOnnx.imageToBitmap(image);
            // image -> bitmap
            long startTime = System.currentTimeMillis();
            Bitmap bitmap_640 = supportOnnx.letterboxResize(image);

            // bitmap -> float buffer
            FloatBuffer imgDataFloat = supportOnnx.bitmapToFloatBuffer(bitmap_640);

            //모델명
            String inputName = ortSession.getInputNames().iterator().next();
            //모델의 요구 입력값
            long[] shape = {SupportOnnx.BATCH_SIZE, SupportOnnx.PIXEL_SIZE, SupportOnnx.INPUT_SIZE, SupportOnnx.INPUT_SIZE};

            try {
                // float buffer -> tensor
                OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnvironment, imgDataFloat, shape);
                // 추론
                OrtSession.Result result = ortSession.run(Collections.singletonMap(inputName, inputTensor));
                // 결과 (v8 의 출력은 [1][xywh + label 의 개수][8400] 입니다.
                float[][][] output = (float[][][]) result.get(0).getValue();

                int rows = output[0][0].length; //8400
                // tensor -> label, score, rectF
                ArrayList<Result> results = supportOnnx.outputsToNMSPredictions(output, rows);
                if (results.size()>0){
                    textView.setText("region: "+results.get(0).getRectF().toString());
                    textView.setTextColor(Color.BLACK);
                }
                System.out.println("yolo detect result: "+results.size());
                DectectResult dectectResult = supportOnnx.TransformDetectResult(results);
                if (dectectResult!=null) {
                    System.out.println("over detect result:"+dectectResult.getOver_status());
                    String type = dectectResult.getDelivery_catagory();
                    overtv.setText("over lap: "+dectectResult.getOver_status());
                }else {
                    System.out.println("detect result is null, skip it");
                    overtv.setText("");
                }
                overtv.setTextColor(Color.BLACK);
                long endTime = System.currentTimeMillis();

                // Calculate execution time
                long executionTime = endTime - startTime;
                System.out.println("detect Execution time: " + executionTime + " milliseconds");
                return results;

            } catch (OrtException e) {
                e.printStackTrace();

            }
        }
        return null;
    }


    public void load_model() {
        //model, label 불러오기
        supportOnnx = new SupportOnnx(this);
        supportOnnx.loadModel();
        supportOnnx.loadLabel();
        try {
            //onnxRuntime 활성화
            ortEnvironment = OrtEnvironment.getEnvironment();
            ortSession = ortEnvironment.createSession(this.getFilesDir().getAbsolutePath() + "/" + SupportOnnx.fileName,
                    new OrtSession.SessionOptions());
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }


}
