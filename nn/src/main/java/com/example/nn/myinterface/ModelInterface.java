package com.example.nn.myinterface;

import com.example.nn.model.modelrun.BaseRun;

public interface ModelInterface {
//    运行完成一次decoder返回的字符串
    public void onDecoder(String decoder);
//   解码完成
    public void onDecodEnd();
//   模型加载完成
    public BaseRun onLoadModel();
//   模型加载失败
    public void onLoadFaile(String str,Exception e);

}
