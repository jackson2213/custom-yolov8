package com.tencent.yolov8ncnn.yolov8;



public class DectectResult {
    private String delivery_catagory;
    private String over_status;


    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    private int label;



    public String getDelivery_catagory() {
        return delivery_catagory;
    }

    public void setDelivery_catagory(String delivery_catagory) {
        this.delivery_catagory = delivery_catagory;
    }

    public String getOver_status() {
        return over_status;
    }

    public void setOver_status(String over_status) {
        this.over_status = over_status;
    }



}
