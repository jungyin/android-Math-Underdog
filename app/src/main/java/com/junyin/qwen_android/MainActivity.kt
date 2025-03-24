package com.example.qwen_android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.provider.Settings
import android.util.Log
import android.widget.Button
import android.widget.EditText
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.qwen_android.base.BaseActivity
import java.io.File


class MainActivity : BaseActivity() {


    var imgpath = "http://upload.jianshu.io/images/js-qrc.png"


    var adapter = LLamaAdapter()
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        adapter.setContext(this)

        loadPermission()

        intoView()


    }

    private fun loadPermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_DENIED
        ) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(
                    arrayOf(
                        Manifest.permission.CAMERA,
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                    ), 11
                )
            }
        } else {
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (!Environment.isExternalStorageManager()) {
                val intent = Intent(
                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION
                )
                intent.setData(Uri.parse("package:$packageName"))
                startActivity(intent)
            }
        }

    }

    private fun intoView() {
//        var f: File = File("/storage/emulated/0/Download/test.png")
        var f: File = File("/storage/emulated/0/Download/1.jpg")
        imgpath = "https://avatars.githubusercontent.com/u/12974087?s=64&v=4"
        imgpath = f.path

        val gifTest: String =
            "<h3>Test9912931293192931932123</h3><img src=\"" + imgpath + "\" /><h3>Test9912931293192931932123</h3><img src=\"" + imgpath + "\" />"

        val rv = findViewById<RecyclerView>(R.id.rv)
        rv.adapter = adapter
        rv.layoutManager = LinearLayoutManager(this, LinearLayoutManager.VERTICAL, false)

        var list = ArrayList<LLamaText>()
        list.add(LLamaText("只是测试文字", LLamaText.chatType))
        list.add(
            LLamaText(
                "\$\$\n" +
                        "R_{e}=\\frac{\\rho v D^{2}} {u}\n" +
                        "\$\$", LLamaText.chatType
            )
        )
        list.add(LLamaText(gifTest, LLamaText.peopleType))

        var et = findViewById<EditText>(R.id.input_et)

        findViewById<Button>(R.id.send_btn).setOnClickListener {
            list.add(LLamaText(et.text.toString(), LLamaText.peopleType))
            adapter.upDate(list)
            rv.scrollToPosition(list.size-1)
        };

        adapter.upDate(list)


    }

}

