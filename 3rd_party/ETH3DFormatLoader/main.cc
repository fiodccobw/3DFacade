// Copyright 2017 Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstring>
#include <iostream>

#include "cameras.h"
#include "images.h"


int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <path_to_cameras_txt> <path_to_images_txt>" << std::endl;
    return EXIT_FAILURE;
  }
  
  const char* cameras_txt_path = argv[1];
  const char* images_txt_path = argv[2];
  
  // Load cameras (indexed by: camera_id).
  ColmapCameraPtrMap cameras;
  bool success = ReadColmapCameras(cameras_txt_path, &cameras);
  if (success) {
    std::cout << "Successfully loaded " << cameras.size() << " camera(s)." << std::endl;
  } else {
    std::cout << "Error: could not load cameras." << std::endl;
  }
  
  // Load images (indexed by: image_id).
  ColmapImagePtrMap images;
  success = ReadColmapImages(images_txt_path, /* read_observations */ true,
                             &images);
  if (success) {
    std::cout << "Successfully loaded " << images.size() << " image info(s)." << std::endl;
  } else {
    std::cout << "Error: could not load image infos." << std::endl;
  }
  
  return EXIT_SUCCESS;
}
