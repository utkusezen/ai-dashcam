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
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import java.io.IOException
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.ui.layout.ContentScale

data class ApiResult(
    val recommendedSpeed: Int,
    val signs: List<Int>
)

private val speedLimitMap = mapOf(
    0 to 5,
    1 to 15,
    2 to 30,
    3 to 40,
    4 to 50,
    5 to 60,
    6 to 70,
    7 to 80,
    58 to 20,
    60 to 100,
    61 to 120
)

class MainActivity : ComponentActivity() {
    private var apiBaseUrl by mutableStateOf("10.0.2.2")
    private var lastApiResult by mutableStateOf<ApiResult?>(null)

    // main | auto | single
    private var currentScreen by mutableStateOf("main")
    private val REQUEST_INTERVAL_MS = 1000L
    private var lastRequestStartTime = 0L
    @Volatile private var isRequestRunning = false
    private lateinit var cameraExecutor: java.util.concurrent.ExecutorService
    private var isAutoRunning = false
    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            if (isGranted) {
                startCamera()
            } else {
                Log.e("CAMERA", "Permission denied")
            }
        }
    private val pickImageLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            uri?.let {
                currentScreen = "single"
                sendImageToBackend(this, it)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

        enableEdgeToEdge()

        setContent {
            AidashcamTheme {

                when (currentScreen) {

                    "main" -> {
                        MainScreen(
                            apiBaseUrl = apiBaseUrl,
                            onApiChange = { apiBaseUrl = it },
                            onPickImage = {
                                currentScreen = "single"
                                pickImageLauncher.launch(arrayOf("image/*"))
                            },
                            onStartAuto = {
                                currentScreen = "auto"
                                checkCameraPermissionAndStart()
                            }
                        )
                    }

                    "auto" -> {
                        ResultScreen(
                            result = lastApiResult,
                            isAutoMode = true,
                            onBack = {
                                stopCamera()
                                currentScreen = "main"
                            }
                        )
                    }

                    "single" -> {
                        ResultScreen(
                            result = lastApiResult,
                            isAutoMode = false,
                            onBack = {
                                currentScreen = "main"
                            }
                        )
                    }
                }
            }
        }
    }

    private fun toggleAutoMode() {
        if (isAutoRunning) {
            stopCamera()
        } else {
            startCamera()
        }
        isAutoRunning = !isAutoRunning
    }

    private fun checkCameraPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture =
            androidx.camera.lifecycle.ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val imageAnalysis = androidx.camera.core.ImageAnalysis.Builder()
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->

                val now = System.currentTimeMillis()

                if (!isRequestRunning &&
                    now - lastRequestStartTime >= REQUEST_INTERVAL_MS
                ) {

                    lastRequestStartTime = now
                    isRequestRunning = true

                    val jpegBytes = imageProxyToJpeg(imageProxy)
                    sendBytesToBackend(jpegBytes)
                }

                imageProxy.close()
            }

            val cameraSelector =
                androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    private fun stopCamera() {
        val cameraProviderFuture =
            androidx.camera.lifecycle.ProcessCameraProvider.getInstance(this)
        val cameraProvider = cameraProviderFuture.get()
        cameraProvider.unbindAll()
    }

    private fun imageProxyToJpeg(image: ImageProxy): ByteArray {

        val width = image.width
        val height = image.height

        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val yPixelStride = yPlane.pixelStride
        val uRowStride = uPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val vPixelStride = vPlane.pixelStride

        val nv21 = ByteArray(width * height * 3 / 2)

        var pos = 0
        for (row in 0 until height) {
            val rowStart = row * yRowStride
            for (col in 0 until width) {
                nv21[pos++] = yBuffer.get(rowStart + col * yPixelStride)
            }
        }

        val uvHeight = height / 2
        val uvWidth = width / 2

        for (row in 0 until uvHeight) {
            val uRowStart = row * uRowStride
            val vRowStart = row * vRowStride

            for (col in 0 until uvWidth) {
                val uIndex = uRowStart + col * uPixelStride
                val vIndex = vRowStart + col * vPixelStride

                nv21[pos++] = vBuffer.get(vIndex)
                nv21[pos++] = uBuffer.get(uIndex)
            }
        }

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            width,
            height,
            null
        )

        val jpegOut = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, width, height),
            90,
            jpegOut
        )

        val jpegBytes = jpegOut.toByteArray()

        val rotationDegrees = image.imageInfo.rotationDegrees
        if (rotationDegrees == 0) {
            return jpegBytes
        }

        val originalBitmap =
            android.graphics.BitmapFactory.decodeByteArray(
                jpegBytes,
                0,
                jpegBytes.size
            )

        val matrix = android.graphics.Matrix().apply {
            postRotate(rotationDegrees.toFloat())
        }

        val rotatedBitmap = android.graphics.Bitmap.createBitmap(
            originalBitmap,
            0,
            0,
            originalBitmap.width,
            originalBitmap.height,
            matrix,
            true
        )

        val rotatedOut = ByteArrayOutputStream()
        rotatedBitmap.compress(
            android.graphics.Bitmap.CompressFormat.JPEG,
            90,
            rotatedOut
        )

        return rotatedOut.toByteArray()
    }

    private fun sendImageToBackend(context: Context, uri: Uri) {
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
            .url("http://$apiBaseUrl:5000/analyze")
            .post(multipartBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("API", "Request failed", e)
            }

            override fun onResponse(call: Call, response: Response) {
                val body = response.body?.string() ?: return

                runOnUiThread {
                    lastApiResult = parseApiResponse(body)
                }
            }
        })
    }

    private fun sendBytesToBackend(bytes: ByteArray) {

        val requestBody = bytes.toRequestBody("image/jpeg".toMediaType())

        val multipartBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("image", "frame.jpg", requestBody)
            .build()

        val request = Request.Builder()
            .url("http://$apiBaseUrl:5000/analyze")
            .post(multipartBody)
            .build()

        OkHttpClient().newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e("API", "Auto request failed", e)
                isRequestRunning = false
            }

            override fun onResponse(call: Call, response: Response) {
                val body = response.body?.string() ?: return

                runOnUiThread {
                    lastApiResult = parseApiResponse(body)
                }

                isRequestRunning = false
            }
        })
    }

    private fun parseApiResponse(json: String): ApiResult {

        val obj = JSONObject(json)

        val recommended = obj.getDouble("recommended_speed").toInt()

        val signsJson = obj.getJSONArray("signs")
        val signsList = mutableListOf<Int>()

        for (i in 0 until signsJson.length()) {
            signsList.add(signsJson.getInt(i))
        }

        return ApiResult(
            recommendedSpeed = recommended,
            signs = signsList
        )
    }
}

@Composable
fun MainScreen(
    apiBaseUrl: String,
    onApiChange: (String) -> Unit,
    onPickImage: () -> Unit,
    onStartAuto: () -> Unit
) {

    androidx.compose.foundation.layout.Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .padding(24.dp),
        horizontalAlignment = androidx.compose.ui.Alignment.CenterHorizontally,
        verticalArrangement = androidx.compose.foundation.layout.Arrangement.Center
    ) {

        Text("Enter the IP Address of the API")

        androidx.compose.material3.OutlinedTextField(
            value = apiBaseUrl,
            onValueChange = onApiChange,
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp)
        )

        androidx.compose.material3.Button(
            onClick = onPickImage,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Text("Analyze Singular Image")
        }

        androidx.compose.material3.Button(
            onClick = onStartAuto,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Text("Start Automatic Analysis")
        }
    }
}

@Composable
fun ResultScreen(
    result: ApiResult?,
    isAutoMode: Boolean,
    onBack: () -> Unit
) {

    val context = androidx.compose.ui.platform.LocalContext.current
    val finalSpeed = result?.let { calculateFinalSpeed(it) }

    androidx.compose.foundation.layout.Column(
        modifier = Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .padding(horizontal = 16.dp, vertical = 16.dp)
    ) {

        Text(
            text = "Running Analysis...",
            modifier = Modifier.padding(top = 8.dp)
        )

        result?.let {

            val lowerBound = finalSpeed?.minus(10) ?: 0
            val upperBound = finalSpeed ?: 0

            Text(
                text = "$lowerBound - $upperBound km/h",
                style = MaterialTheme.typography.headlineLarge,
                modifier = Modifier
                    .padding(start = 0.dp, top = 16.dp)
            )

            if (it.signs.isNotEmpty()) {

                androidx.compose.foundation.layout.Row(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxWidth()
                ) {

                    it.signs.forEach { signId ->

                        val imageBitmap = remember(signId) {
                            context.assets.open("traffic-signs/$signId.png")
                                .use { input ->
                                    android.graphics.BitmapFactory
                                        .decodeStream(input)
                                        .asImageBitmap()
                                }
                        }

                        androidx.compose.foundation.Image(
                            bitmap = imageBitmap,
                            contentDescription = null,
                            contentScale = ContentScale.Fit,
                            modifier = Modifier
                                .padding(4.dp)
                                .size(72.dp)
                        )
                    }
                }
            }
        }

        androidx.compose.material3.Button(
            onClick = onBack,
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text("Zurück")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    AidashcamTheme {
        //MainScreen("Android")
    }
}

private fun calculateFinalSpeed(result: ApiResult): Int {

    val baseSpeed = result.recommendedSpeed

    val detectedLimits = result.signs
        .mapNotNull { speedLimitMap[it] }

    if (detectedLimits.isEmpty()) return baseSpeed

    val lowestSignLimit = detectedLimits.minOrNull()!!

    return minOf(baseSpeed, lowestSignLimit)
}


fun uriToRequestBody(context: Context, uri: Uri): RequestBody {
    val inputStream = context.contentResolver.openInputStream(uri)
        ?: throw IllegalArgumentException("Cannot open input stream")

    val bytes = inputStream.readBytes()
    inputStream.close()

    return bytes.toRequestBody("image/jpeg".toMediaType())
}

