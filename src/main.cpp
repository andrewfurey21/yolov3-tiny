#include <cstdint>
#include <mlpack.hpp>
#include <armadillo>
#include <stdexcept>
#include <stdio.h>

#include "../models/models/yolov3_tiny/yolov3_tiny.hpp"


#define INPUT "input.jpg"
#define OUTPUT "output.jpg"
#define WEIGHTS "yolov3-tiny.weights"

using namespace mlpack::models;

struct detection {
  double x, y, w, h;
  std::vector<double> classProbabilities;
  double objectness;
};

class Image {
public:
  Image() :
    width(0),
    height(0),
    channels(0)
  {}

  Image(size_t width, size_t height, size_t channels) :
  width(width),
  height(height),
  channels(channels)
  {
    data = std::vector<double>(width * height * channels);
  }

  void load(const char* fileName) {
    int width, height, channels;
    uint8_t* data = stbi_load(fileName, &width, &height, &channels, 3);
    if (!data) throw std::runtime_error("Could not load image\n");

    this->channels = 3;
    this->width = width;
    this->height = height;

    for (int c = 0; c < this->channels; ++c){
        for (int j = 0; j < this->height; ++j){
            for (int i = 0; i < this->width; ++i){
                int dest = i + this->width*j + this->width*this->height*c;
                int source = c + c*i + c*this->width*j;
                this->data[dest] = (double)data[source]/255.0f;
            }
        }
    }
    stbi_image_free(data);
  }

  void save(const char* fileName) {
    uint8_t* data = (uint8_t*)calloc(this->width*this->height*this->channels, sizeof(uint8_t)); 
    for (int c = 0; c < this->channels; ++c) {
        for (int i = 0; i < this->width*this->height; ++i) {
            data[i*this->channels+c] = (uint8_t)(255*this->data[i + c*this->width*this->height]);
        }
    }
    int success = stbi_write_jpg(fileName, this->width, this->height, this->channels, data, 80);
    if (!success) {
	throw std::runtime_error("Could not write to image\n");
    }
    free(data);
  }

  void embed(Image& source, Image& dest, size_t dx, size_t dy) {
    for (size_t c = 0; c < source.channels; c++) {
      for (size_t y = 0; y < source.height; y++) {
        for (size_t x = 0; x < source.width; x++) {
          double value = source.getPixel(x, y, c);
          dest.setPixel(x+dx, y+dy, c, value);
        }
      }
    }
  }

  void resize(Image& source, Image& dest, size_t newWidth, size_t newHeight) {
    dest.data = std::vector<double>(newWidth * newHeight * source.channels);
    //TODO: interpolate across x and y
  }

  Image letterboxImage(size_t newWidth, size_t newHeight) {
    int newW = this->width;
    int newH = this->height;
    if (((float)newWidth/newW) < ((float)newHeight/newH)) {
        newW = newWidth;
        newH = this->height * newWidth / this->width;
    } else {
        newH = newHeight;
        newW = this->width * newHeight / this->height;
    }
    Image resized;
    resize(*this, resized, newW, newH);
    Image boxed(newWidth, newHeight, this->channels);
    boxed.fill(0.5);//fill with gray
    embed(resized, boxed, (newWidth - newW)/2, (newHeight - newH)/2);
    return boxed;
  }

  void drawBox(size_t x, size_t y, size_t w, size_t h, size_t border) {
    if (x < 0) x = 0;
    if (x >= this->width) x = this->width - 1;

    if (y < 0) y = 0;
    if (y >= this->height) y = this->height - 1;

    if (x+w > this->width) w = this->width - x;
    if (x+w > this->width) w = this->width - x;

    if (y+h > this->height) h = this->height - y;
    if (y+h > this->height) h = this->height - y;

    for (size_t b = 0; b < border; b++) {
      for (size_t i = x; i < w; i++) {
        for (size_t j = y; j < h; j++) {
          setPixel(i, j, 0, 1.0f);//TODO:dont fill
          setPixel(i, j, 1, 1.0f);
          setPixel(i, j, 2, 1.0f);
        }
      }
    }
  }

  void setPixel(size_t x, size_t y, size_t c, double value) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < width);
    assert(c >= 0 && c < channels);

    size_t index = c * width * height + y * width + x;
    data[index] = value;
  }

  double getPixel(size_t x, size_t y, size_t c) {
    assert(x >= 0 && x < width);
    assert(y >= 0 && y < width);
    assert(c >= 0 && c < channels);

    size_t index = c * width * height + y * width + x;
    return data[index];
  }

  void fill(double value) {
    std::fill(data.begin(), data.end(), value);
  }

private:
  size_t width, height, channels;
  std::vector<double> data;
};

int main(void) {
  return 0;
}
