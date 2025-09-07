#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
// ====== CONFIGURE THESE ======
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASS = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "http://<LAPTOP_IP>:8000/predict"; // Replace <LAPTOP_IP>
// Relay pin (through transistor/relay module). Many boards use GPIO 12/13/14/15.
const int RELAY_PIN = 12; // Adjust as wired; LOW-active modules may be common.
// Decision mapping: which label triggers spray
// Expected labels file contains: Healthy, Mild, Severe (order may vary)
String SPRAY_ON_LABEL = "Severe"; // Spray when Severe
float  SPRAY_THRESHOLD = 0.60;    // Minimum confidence to spray
// ==============================
// AI Thinker ESP32-CAM pinout
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
void setupCamera() {
	camera_config_t config;
	config.ledc_channel = LEDC_CHANNEL_0;
	config.ledc_timer = LEDC_TIMER_0;
	config.pin_d0 = Y2_GPIO_NUM;
	config.pin_d1 = Y3_GPIO_NUM;
	config.pin_d2 = Y4_GPIO_NUM;
	config.pin_d3 = Y5_GPIO_NUM;
	config.pin_d4 = Y6_GPIO_NUM;
	config.pin_d5 = Y7_GPIO_NUM;
	config.pin_d6 = Y8_GPIO_NUM;
	config.pin_d7 = Y9_GPIO_NUM;
	config.pin_xclk = XCLK_GPIO_NUM;
	config.pin_pclk = PCLK_GPIO_NUM;
	config.pin_vsync = VSYNC_GPIO_NUM;
	config.pin_href = HREF_GPIO_NUM;
	config.pin_sscb_sda = SIOD_GPIO_NUM;
	config.pin_sscb_scl = SIOC_GPIO_NUM;
	config.pin_pwdn = PWDN_GPIO_NUM;
	config.pin_reset = RESET_GPIO_NUM;
	config.xclk_freq_hz = 20000000;
	config.pixel_format = PIXFORMAT_JPEG;

	// VGA is a good trade-off size/speed
	config.frame_size = FRAMESIZE_VGA; // 640x480
	config.jpeg_quality = 12;          // 0-63 (lower is better quality)
	config.fb_count = 2;

	esp_err_t err = esp_camera_init(&config);
	if (err != ESP_OK) {
		Serial.printf("Camera init failed with error 0x%x\n", err);
		while (true) { delay(1000); }
	}
}
void setup() {
	Serial.begin(115200);
	pinMode(RELAY_PIN, OUTPUT);
	digitalWrite(RELAY_PIN, HIGH); // Assume HIGH = off for active-low relay

	setupCamera();

	WiFi.begin(WIFI_SSID, WIFI_PASS);
	Serial.println("Connecting to WiFi...");
	while (WiFi.status() != WL_CONNECTED) {
		delay(500);
		Serial.print(".");
	}
	Serial.println("\nWiFi connected.");
	Serial.print("IP: "); Serial.println(WiFi.localIP());
}

String classifyAndMaybeSpray(const uint8_t* jpg, size_t len) {
	HTTPClient http;
	WiFiClient client;

	http.begin(client, SERVER_URL);

	String boundary = "------------------------esp32camBoundary";
	String contentType = "multipart/form-data; boundary=" + boundary;

	// Build multipart body in memory (simple, not streaming)
	String head = "--" + boundary + "\r\n";
	head += "Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n";
	head += "Content-Type: image/jpeg\r\n\r\n";
	String tail = "\r\n--" + boundary + "--\r\n";

	int totalLen = head.length() + len + tail.length();

	http.addHeader("Content-Type", contentType);
	http.addHeader("Connection", "keep-alive");

	int rc = http.sendRequest("POST", (uint8_t*)NULL, 0); // Prepare connection
	if (rc <= 0) {
		http.end();
		return String("HTTP connect failed");
	}

	WiFiClient* stream = http.getStreamPtr();
	stream->print(head);
	stream->write(jpg, len);
	stream->print(tail);

	int status = http.getSize();
	(void)status;

	int code = http.responseStatusCode();
	String payload = http.getString();
	http.end();

	Serial.printf("HTTP %d\n", code);
	Serial.println(payload);

	bool spray = false;
	if (code == 200) {
		// Very naive JSON parsing: search for label and probability
		int lpos = payload.indexOf("\"label\":\"");
		if (lpos >= 0) {
			int lstart = lpos + 9;
			int lend = payload.indexOf("\"", lstart);
			String label = payload.substring(lstart, lend);

			int ppos = payload.indexOf("\"probability\":");
			float prob = 0.0;
			if (ppos >= 0) {
				int pstart = ppos + 15;
				int pend = payload.indexOf(",", pstart);
				if (pend < 0) pend = payload.indexOf("}\n", pstart);
				if (pend < 0) pend = payload.length();
				String pstr = payload.substring(pstart, pend);
				prob = pstr.toFloat();
			}

			if (label == SPRAY_ON_LABEL && prob >= SPRAY_THRESHOLD) {
				spray = true;
			}
		}
	}

	if (spray) {
		Serial.println("Spray ON");
		digitalWrite(RELAY_PIN, LOW); // active-low relay ON
		delay(1500);                  // spray duration
		digitalWrite(RELAY_PIN, HIGH);
	} else {
		Serial.println("Spray OFF");
		digitalWrite(RELAY_PIN, HIGH);
	}

	return payload;
}

void loop() {
	// Capture a frame
	camera_fb_t * fb = esp_camera_fb_get();
	if (!fb) {
		Serial.println("Camera capture failed");
		delay(2000);
		return;
	}

	if (fb->format != PIXFORMAT_JPEG) {
		Serial.println("Non-JPEG frame; skipping");
		esp_camera_fb_return(fb);
		delay(1000);
		return;
	}

	classifyAndMaybeSpray(fb->buf, fb->len);

	esp_camera_fb_return(fb);

	delay(5000); // wait before next capture
}