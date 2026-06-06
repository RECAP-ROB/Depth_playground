#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include "esp_http_server.h"

// --- Network Credentials ---
const char* ssid = "Phyrocks";
const char* password = "Tenfatpigs10";

// --- Stereo Identity (Managed by PlatformIO build_flags) ---
#ifndef CAMERA_ID
  #define CAMERA_ID 0
#endif

#if CAMERA_ID == 0
  #define MDNS_HOSTNAME "stereo-left"
  #define CAM_SIDE      "LEFT"
#else
  #define MDNS_HOSTNAME "stereo-right"
  #define CAM_SIDE      "RIGHT"
#endif

#define PART_BOUNDARY "123456789000000000000987654321"

// --- Hardware Pin Mappings ---
#if defined(CAMERA_MODEL_AI_THINKER)
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

#elif defined(CAMERA_MODEL_ESP32S3_EYE)
  #define PWDN_GPIO_NUM    -1
  #define RESET_GPIO_NUM   -1
  #define XCLK_GPIO_NUM    15
  #define SIOD_GPIO_NUM     4
  #define SIOC_GPIO_NUM     5
  #define Y9_GPIO_NUM      16
  #define Y8_GPIO_NUM      17
  #define Y7_GPIO_NUM      18
  #define Y6_GPIO_NUM      12
  #define Y5_GPIO_NUM      10
  #define Y4_GPIO_NUM       8
  #define Y3_GPIO_NUM       9
  #define Y2_GPIO_NUM      11
  #define VSYNC_GPIO_NUM    6
  #define HREF_GPIO_NUM     7
  #define PCLK_GPIO_NUM    13
#endif

static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

httpd_handle_t stream_httpd = NULL;
char cachedIP[16] = {0};

void cacheIP() {
  strncpy(cachedIP, WiFi.localIP().toString().c_str(), sizeof(cachedIP));
}

static esp_err_t stream_handler(httpd_req_t *req){
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char * part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if(res != ESP_OK) return res;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while(true){
    fb = esp_camera_fb_get();
    if (!fb) {
      res = ESP_FAIL;
    } else {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
      if(res == ESP_OK){
        size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, fb->len);
        res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
      }
      if(res == ESP_OK){
        res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      }
      esp_camera_fb_return(fb);
    }
    if(res != ESP_OK) break;
    vTaskDelay(pdMS_TO_TICKS(1));
  }
  return res;
}

void startCameraServer(){
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

void printConnectionInfo() {
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.printf("║  Camera  : %-28s║\n", CAM_SIDE);
  Serial.printf("║  mDNS    : http://%-19s.local ║\n", MDNS_HOSTNAME);
  Serial.printf("║  Stream  : http://%-20s/stream ║\n", cachedIP);
  Serial.println("╚════════════════════════════════════════╝\n");
}

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);

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

  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  if (esp_camera_init(&config) != ESP_OK) return;

  sensor_t * s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_hmirror(s, 0);

  // Consistent imaging settings for stereo pair
  s->set_whitebal(s, 0);       // disable auto white balance
  s->set_awb_gain(s, 0);       // disable AWB gain
  s->set_wb_mode(s, 0);        // auto WB mode off
  s->set_exposure_ctrl(s, 0);  // disable auto exposure
  s->set_gain_ctrl(s, 0);      // disable auto gain
  s->set_aec_value(s, 800);    // same exposure on both cameras (0-1200)
  s->set_agc_gain(s, 8);       // same gain on both cameras (0-30)
  s->set_bpc(s, 1);            // enable bad pixel correction
  s->set_wpc(s, 1);            // enable white pixel correction

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  cacheIP();

  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("\nmDNS started: http://%s.local\n", MDNS_HOSTNAME);
  }

  startCameraServer();
  printConnectionInfo();
}

void loop() {
  delay(10000);
}
