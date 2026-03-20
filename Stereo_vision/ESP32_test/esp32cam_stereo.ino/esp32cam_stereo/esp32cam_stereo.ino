/**
 * ESP32-CAM Stereo Vision Node
 * ============================
 * Flash this to BOTH ESP32-CAMs.
 * Set CAMERA_ID to 0 for LEFT camera, 1 for RIGHT camera.
 * Both cameras must be on the same WiFi network.
 *
 * The host (PC or Raspberry Pi 4B) connects to each camera's
 * HTTP stream and processes the stereo pair.
 *
 * Hardware: AI-Thinker ESP32-CAM (OV2640)
 * Board:    "AI Thinker ESP32-CAM" in Arduino IDE
 *
 * Libraries required:
 *   - ESP32 Arduino core (espressif/arduino-esp32)
 *   - ESPmDNS (included in ESP32 Arduino core — no extra install needed)
 *
 * After flashing, open Serial Monitor at 115200 baud.
 * The camera prints its IP address AND its mDNS hostname, e.g.:
 *
 *   LEFT  camera → http://stereo-left.local/stream
 *   RIGHT camera → http://stereo-right.local/stream
 *
 * Use the .local hostnames in stereo_processor.py so you never
 * need to chase DHCP addresses again.
 * NOTE: mDNS (.local) requires the host to be on the same WiFi subnet.
 *       Works on Linux/macOS natively. On Windows install Bonjour:
 *       https://support.apple.com/kb/DL999
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_http_server.h"
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"

// ─────────────────────────────────────────────
//  USER CONFIGURATION — edit these values
// ─────────────────────────────────────────────
#define CAMERA_ID     0           // 0 = LEFT camera, 1 = RIGHT camera
#define WIFI_SSID     "NAKUJA"
#define WIFI_PASSWORD "987654321"

// mDNS hostnames (no need to change these)
//   LEFT  → http://stereo-left.local/stream
//   RIGHT → http://stereo-right.local/stream
const char* MDNS_NAME = (CAMERA_ID == 0) ? "stereo-left" : "stereo-right";

// ─────────────────────────────────────────────
//  AI-Thinker ESP32-CAM pin map
// ─────────────────────────────────────────────
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22
#define LED_GPIO_NUM     4   // onboard flash LED (active HIGH)

// ─────────────────────────────────────────────
//  Stream part boundary
// ─────────────────────────────────────────────
#define PART_BOUNDARY "frame"
static const char* _STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART =
    "Content-Type: image/jpeg\r\n"
    "Content-Length: %u\r\n"
    "X-Camera-ID: %d\r\n\r\n";

httpd_handle_t stream_httpd = NULL;

// ─────────────────────────────────────────────
//  MJPEG stream handler
// ─────────────────────────────────────────────
static esp_err_t stream_handler(httpd_req_t* req) {
  camera_fb_t* fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  // Disable response buffering for lower latency
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Camera-ID", CAMERA_ID == 0 ? "LEFT" : "RIGHT");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    size_t hlen = snprintf(part_buf, sizeof(part_buf), _STREAM_PART,
                           fb->len, CAMERA_ID);

    res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY,
                                strlen(_STREAM_BOUNDARY));
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK)
      res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);

    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK) break;
  }
  return res;
}

// ─────────────────────────────────────────────
//  Simple status page
// ─────────────────────────────────────────────
static esp_err_t index_handler(httpd_req_t* req) {
  char html[512];
  snprintf(html, sizeof(html),
    "<html><body style='font-family:monospace'>"
    "<h2>ESP32-CAM Stereo Node</h2>"
    "<p>Camera ID: <b>%s</b></p>"
    "<p>Stream: <a href='/stream'>/stream</a></p>"
    "<p>IP: %s</p>"
    "</body></html>",
    CAMERA_ID == 0 ? "LEFT (0)" : "RIGHT (1)",
    WiFi.localIP().toString().c_str());
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_sendstr(req, html);
}

// ─────────────────────────────────────────────
//  Start HTTP server
// ─────────────────────────────────────────────
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 4;

  httpd_uri_t index_uri = {
    .uri      = "/",
    .method   = HTTP_GET,
    .handler  = index_handler,
    .user_ctx = NULL
  };
  httpd_uri_t stream_uri = {
    .uri      = "/stream",
    .method   = HTTP_GET,
    .handler  = stream_handler,
    .user_ctx = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("HTTP server started");
  }
}

// ─────────────────────────────────────────────
//  Setup
// ─────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false);

  // Turn off flash LED
  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Use PSRAM if available for larger frames
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_VGA;   // 640×480
    config.jpeg_quality = 12;              // 0–63, lower = better quality
    config.fb_count     = 2;
  } else {
    config.frame_size   = FRAMESIZE_QVGA;  // 320×240
    config.jpeg_quality = 15;
    config.fb_count     = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return;
  }

  // Fine-tune sensor settings for better stereo matching
  sensor_t* s = esp_camera_sensor_get();
  s->set_brightness(s, 0);      // -2 to 2
  s->set_contrast(s, 0);        // -2 to 2
  s->set_saturation(s, 0);      // -2 to 2
  s->set_sharpness(s, 0);       // -2 to 2
  s->set_whitebal(s, 1);        // auto white balance ON
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0);         // 0=auto
  s->set_exposure_ctrl(s, 1);   // auto exposure ON
  s->set_aec2(s, 0);
  s->set_gain_ctrl(s, 1);       // auto gain ON
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)0);
  s->set_bpc(s, 0);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_hmirror(s, CAMERA_ID); // Mirror RIGHT camera for correct stereo orientation
  s->set_vflip(s, 0);
  s->set_dcw(s, 1);
  s->set_colorbar(s, 0);

  // ── WiFi (DHCP) ──────────────────────────────
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);  // disable power-save for stable streaming

  Serial.printf("\nConnecting to '%s'", WIFI_SSID);
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    // Blink LED while connecting
    digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
    if (millis() - t0 > 30000) {
      Serial.println("\nWiFi timeout — restarting…");
      ESP.restart();
    }
  }
  digitalWrite(LED_GPIO_NUM, LOW);  // LED off once connected
  Serial.println(" connected!\n");

  // ── mDNS ─────────────────────────────────────
  if (MDNS.begin(MDNS_NAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("mDNS started: http://%s.local\n", MDNS_NAME);
  } else {
    Serial.println("mDNS failed — use IP address instead");
  }

  // ── Print connection info ─────────────────────
  const char* side = (CAMERA_ID == 0) ? "LEFT" : "RIGHT";
  Serial.println("════════════════════════════════════════");
  Serial.printf("  Camera  : %s (ID %d)\n", side, CAMERA_ID);
  Serial.printf("  IP addr : http://%s/stream\n",
                WiFi.localIP().toString().c_str());
  Serial.printf("  mDNS    : http://%s.local/stream\n", MDNS_NAME);
  Serial.println("════════════════════════════════════════\n");

  startCameraServer();
}

// ─────────────────────────────────────────────
//  Loop — nothing to do, server runs on its own
// ─────────────────────────────────────────────
void loop() {
  // Re-print IP every 30 s in case you missed it at boot
  static unsigned long last_print = 0;
  if (millis() - last_print > 30000) {
    last_print = millis();
    Serial.printf("[%s] IP: %s  mDNS: http://%s.local/stream\n",
                  CAMERA_ID == 0 ? "LEFT" : "RIGHT",
                  WiFi.localIP().toString().c_str(),
                  MDNS_NAME);
  }

  // Reconnect if WiFi drops
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost — reconnecting…");
    WiFi.reconnect();
    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 15000)
      delay(500);
    if (WiFi.status() == WL_CONNECTED)
      Serial.printf("Reconnected: %s\n", WiFi.localIP().toString().c_str());
    else
      ESP.restart();
  }

  delay(1000);
}
