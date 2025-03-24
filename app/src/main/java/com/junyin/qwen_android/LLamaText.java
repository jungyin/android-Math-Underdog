package com.junyin.qwen_android;

public class LLamaText {
    String text = "";

    //类型，1代码用户，2代表机器
    protected int type = 0;
    public static final int peopleType = 1;
    public static final int chatType = 2;

    public LLamaText(String text, int type) {
        this.text = text;
        this.type = type;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }
}
