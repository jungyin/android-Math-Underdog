package com.example.qwen_android.base;

import android.app.Application;

import com.zzhoujay.richtext.RichText;

public class BaseApplication  extends Application {
    @Override
    public void onCreate() {
        super.onCreate();

        RichText.initCacheDir(this);
    }
}
