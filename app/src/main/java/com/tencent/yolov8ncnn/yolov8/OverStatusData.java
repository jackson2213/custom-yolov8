package com.tencent.yolov8ncnn.yolov8;

import java.util.PropertyResourceBundle;

public class OverStatusData {
    private static OverStatusData instance;
    private float xmin;
    private float xmax;
    private float xmin_boundary;
    private float xmax_boundary;
    private float centerx;

    private OverStatusData() {
        // Set default values
        xmin = 225;
        xmax = 412;
        xmin_boundary = 25;
        xmax_boundary = 600;
    }

    public static OverStatusData getInstance() {
        if (instance == null) {
            synchronized (OverStatusData.class) {
                if (instance == null) {
                    instance = new OverStatusData();
                }
            }
        }
        return instance;
    }

    public void setOffset(float offset) {
        System.out.println("offset"+offset);
//        if (Math.abs(offset)>10) {
//            this.xmin += offset;
//            this.xmax += offset;
//            this.xmin_boundary += offset;
//            this.xmax_boundary += offset;
//        }

    }

    public double getOverStatus(float box_xmin, float box_xmax){
        if (box_xmin > this.xmin || box_xmax < this.xmax)
            return  0.1;
        else if (box_xmin<this.xmin_boundary ||box_xmax> this.xmax_boundary)
            return 1;
        else if(((box_xmin < this.xmin) && (box_xmin>this.centerx/2)) || (box_xmax>this.xmax && box_xmax< (this.centerx+this.centerx/2))) {
            float rate=this.centerx / 2;
            float overx1 = (this.xmin - box_xmin) / rate;
            float overx2 = (box_xmax-this.xmax)/rate;
            return Math.max(overx1, overx2);
        }else {
            float rate = this.centerx/2-this.xmin_boundary;
            float overx1 = (this.centerx/2 - box_xmin) / rate;
            float overx2 = (box_xmax-(this.centerx+this.centerx/2))/rate;
            return Math.max(overx1, overx2);
        }

    }

    public float getCenterx() {
        return centerx;
    }

    public void setCenterx(float centerx) {
        this.centerx = centerx;
        this.xmin=centerx-centerx/4;
        this.xmax=centerx+centerx/4;
        this.xmin_boundary=centerx/5;
        this.xmax_boundary=centerx*2-centerx/5;
    }

    // Getters and setters for the values
    public float getXmin() {
        return xmin;
    }

    public void setXmin(float xmin) {
        this.xmin = xmin;
    }

    public float getXmax() {
        return xmax;
    }

    public void setXmax(float xmax) {
        this.xmax = xmax;
    }

    public float getXminBoundary() {
        return xmin_boundary;
    }

    public void setXminBoundary(float xmin_boundary) {
        this.xmin_boundary = xmin_boundary;
    }

    public float getXmaxBoundary() {
        return xmax_boundary;
    }

    public void setXmaxBoundary(float xmax_boundary) {
        this.xmax_boundary = xmax_boundary;
    }
}