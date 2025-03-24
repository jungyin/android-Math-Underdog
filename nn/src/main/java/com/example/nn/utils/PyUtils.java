package com.example.nn.utils;

import android.content.Context;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

public class PyUtils {

    public static PyObject pyDetect;

    /***
     * 初始化py
     ***/
    public static void initpy(Context mContext) {
        try {
            //初始化python环境
            if (!Python.isStarted()) {
                Python.start(new AndroidPlatform(mContext));
            }
            Python python = Python.getInstance();
            pyDetect = python.getModule("runpy");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

}
