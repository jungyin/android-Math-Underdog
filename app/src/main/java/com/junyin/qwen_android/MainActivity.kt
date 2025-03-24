package com.junyin.qwen_android

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.widget.Button
import android.widget.EditText
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.junyin.qwen_android.base.BaseActivity
import java.io.File
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Paths


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


        var sb = StringBuffer()
        var content="";
        val filePath = "/storage/emulated/0/Download/testtext.txt" // 替换为你的文件路径
        try {


            // 以字节数组的形式读取整个文件
            val fileBytes = Files.readAllBytes(Paths.get(filePath))


            // 将字节数组转换为字符串，这里使用标准的UTF-8编码来避免转义处理问题
             content = String(fileBytes, StandardCharsets.UTF_8)

            content = content.substring(1, content.length - 1);

//            sb.append("<h3>")
            content = content.replace("\\n","\n")
            content = content.replace("\\t","\t")
            sb.append(content)
            sb.append("\n")
            sb.append("这是第一行\n这是第二行")
//            sb.append("</h3>")
        } catch (e: IOException) {
            e.printStackTrace()
        }

        content = sb.toString()

        list.add(LLamaText(content, LLamaText.chatType))
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

