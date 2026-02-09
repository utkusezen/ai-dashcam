package com.dashcam

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.dashcam.ui.theme.AidashcamTheme
import okhttp3.Call
import okhttp3.Callback
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.File
import java.io.IOException

class MainActivity : ComponentActivity() {
    private val pickImageLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? -> uri?.let { sendImageToBackend(this, it) }}
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pickImageLauncher.launch(arrayOf("image/*"))
        enableEdgeToEdge()
        setContent {
            AidashcamTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    AidashcamTheme {
        Greeting("Android")
    }
}

fun sendImageToBackend(context: Context, uri: Uri) {
    val client = OkHttpClient()
    val requestBody = uriToRequestBody(context, uri)

    val multipartBody = MultipartBody.Builder()
        .setType(MultipartBody.FORM)
        .addFormDataPart(
            name = "image",
            filename = "upload.jpg",
            body = requestBody
        )
        .build()

    val request = Request.Builder()
        .url("http://10.0.2.2:5000/analyze")
        .post(multipartBody)
        .build()

    client.newCall(request).enqueue(object : Callback {
        override fun onFailure(call: Call, e: IOException) {
            Log.e("API", "Request failed", e)
        }

        override fun onResponse(call: Call, response: Response) {
            val body = response.body?.string()
            Log.d("API", "Response: $body")
        }
    })
}

fun uriToRequestBody(context: Context, uri: Uri): RequestBody {
    val inputStream = context.contentResolver.openInputStream(uri)
        ?: throw IllegalArgumentException("Cannot open input stream")

    val bytes = inputStream.readBytes()
    inputStream.close()

    return bytes.toRequestBody("image/jpeg".toMediaType())
}

