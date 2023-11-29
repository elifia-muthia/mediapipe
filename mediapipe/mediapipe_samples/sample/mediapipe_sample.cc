// // Copyright 2019 The MediaPipe Authors.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// // An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <fstream>

// server deps
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

#define PORT 8080

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "landmarks";
constexpr char kWindowName[] = "MediaPipe";
mediapipe::CalculatorGraph graph;

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

//////////////////////
absl::Status initMPPGraph() {
  std::string calculator_graph_config_contents = R"(
    # MediaPipe graph that performs hands tracking on desktop with TensorFlow
    # Lite on CPU.
    # Used in the example in
    # mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu.

    # CPU image. (ImageFrame)
    input_stream: "input_video"

    # CPU image. (ImageFrame)
    output_stream: "output_video"

    # Generates side packet cotaining max number of hands to detect/track.
    node {
      calculator: "ConstantSidePacketCalculator"
      output_side_packet: "PACKET:num_hands"
      node_options: {
        [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
          packet { int_value: 2 }
        }
      }
    }

    # Detects/tracks hand landmarks.
    node {
      calculator: "HandLandmarkTrackingCpu"
      input_stream: "IMAGE:input_video"
      input_side_packet: "NUM_HANDS:num_hands"
      output_stream: "LANDMARKS:landmarks"
      output_stream: "HANDEDNESS:handedness"
      output_stream: "PALM_DETECTIONS:multi_palm_detections"
      output_stream: "HAND_ROIS_FROM_LANDMARKS:multi_hand_rects"
      output_stream: "HAND_ROIS_FROM_PALM_DETECTIONS:multi_palm_rects"
    }

    # Subgraph that renders annotations and overlays them on top of the input
    # images (see hand_renderer_cpu.pbtxt).
    node {
      calculator: "HandRendererSubgraph"
      input_stream: "IMAGE:input_video"
      input_stream: "DETECTIONS:multi_palm_detections"
      input_stream: "LANDMARKS:landmarks"
      input_stream: "HANDEDNESS:handedness"
      input_stream: "NORM_RECTS:0:multi_palm_rects"
      input_stream: "NORM_RECTS:1:multi_hand_rects"
      output_stream: "IMAGE:output_video"
    }

    node {
      calculator: "PacketPresenceCalculator"
      input_stream: "PACKET:landmarks"
      output_stream: "PRESENCE:landmark_presence"
    }
  )";

  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  
  MP_RETURN_IF_ERROR(graph.Initialize(config));
}

absl::Status RunMPPGraph(cv::Mat& inputImage, cv::Mat& outputImage, std::vector<::mediapipe::NormalizedLandmarkList>& multiHandLandmarks) {
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));

  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark,
              graph.AddOutputStreamPoller(kLandmarksStream));

  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller presence_poller,
                   graph.AddOutputStreamPoller("landmark_presence"));

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_RETURN_IF_ERROR(graph.StartRun({}));
  
  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  size_t frame_timestamp = 0;

  cv::Mat inputImgProcessed;
  cv::cvtColor(inputImage, inputImgProcessed, cv::COLOR_BGR2RGB);

  // Wrap Mat into an ImageFrame.
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, inputImgProcessed.cols, inputImgProcessed.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  inputImgProcessed.copyTo(input_frame_mat);

  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
  kInputStream, mediapipe::Adopt(input_frame.release())
                            .At(mediapipe::Timestamp(frame_timestamp++))));

  // Get the graph result packet, or stop if that fails.
  mediapipe::Packet packet;
  mediapipe::Packet landmark_packet; 
  mediapipe::Packet presence_packet;

  if (!poller.Next(&packet)) return absl::Status(absl::StatusCode::kInvalidArgument, "No Output Packet");;

  if (!presence_poller.Next(&presence_packet)) return absl::Status(absl::StatusCode::kInvalidArgument, "No Presence Packet");;
  
  auto is_landmark_present = presence_packet.Get<bool>();

  if (is_landmark_present) {
    if (!poller_landmark.Next(&landmark_packet)) return absl::Status(absl::StatusCode::kInvalidArgument, "No Landmark Packet");;
    multiHandLandmarks = landmark_packet.Get<std::vector<::mediapipe::NormalizedLandmarkList>>();
  }

  auto& output_frame = packet.Get<mediapipe::ImageFrame>();

  // Convert back to opencv for display or saving.
  outputImage = mediapipe::formats::MatView(&output_frame);
  cv::cvtColor(outputImage, outputImage, cv::COLOR_RGB2BGR);

  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  initMPPGraph();

  int serverSocket, clientSocket;
  struct sockaddr_in serverAddress, clientAddress;
  socklen_t clientAddressLength = sizeof(clientAddress);

  // Create socket
  serverSocket = socket(AF_INET, SOCK_STREAM, 0);
  if (serverSocket < 0) {
      std::cerr << "Error creating socket" << std::endl;
      return -1;
  }

  // Set up server address structure
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_addr.s_addr = INADDR_ANY;
  serverAddress.sin_port = htons(PORT);

  // Bind socket to address and port
  if (bind(serverSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) < 0) {
      std::cerr << "Error binding socket" << std::endl;
      close(serverSocket);
      return -1;
  }

  // Listen for incoming connections
  if (listen(serverSocket, 5) < 0) {
      std::cerr << "Error listening for connections" << std::endl;
      close(serverSocket);
      return -1;
  }

  // std::cout << "Server listening on port " << PORT << " ..." << std::endl;
  
  std::cout << " Port: "
              << ntohs(serverAddress.sin_port)
              << std::endl;

  while (true) {
    // Accept a client connection
    clientSocket = accept(serverSocket, (struct sockaddr*)&clientAddress, &clientAddressLength);
    std::cout << "Client connected!" << std::endl;
    if (clientSocket < 0) {
        std::cerr << "Error accepting connection" << std::endl;
        close(serverSocket);
        return -1;
    }

    while (true) {
      // Receive the size of the image data
      size_t imageSize;
      ssize_t sizeReceived = recv(clientSocket, &imageSize, sizeof(size_t), 0);

      if (sizeReceived != sizeof(size_t)) {
          std::cerr << "Error receiving image size" << std::endl;
          close(clientSocket);
          return -1;
      }

      if (imageSize == 0) {
            std::cout << "End of stream signal received. Closing connection." << std::endl;
            break;  // Exit the inner loop and close the connection
      }

      std::cout << "Image size received!\n";

      // Receive the image data
      std::vector<char> imageDataBuffer(imageSize);
      size_t totalBytesReceived = 0;

      while (totalBytesReceived < imageSize) {
          ssize_t bytesRead = recv(clientSocket, imageDataBuffer.data() + totalBytesReceived, imageSize - totalBytesReceived, 0);

          if (bytesRead <= 0) {
              std::cerr << "Error receiving image data" << std::endl;
              close(clientSocket);
              return -1;
          }

          totalBytesReceived += bytesRead;
      }

      cv::Mat inputImage = cv::imdecode(imageDataBuffer, cv::IMREAD_COLOR);

      if (inputImage.empty()) {
          std::cerr << "Error decoding image" << std::endl;
          close(clientSocket);
          return -1;
      }
      cv::imwrite("/Users/elifiamuthia/Desktop/validate_image.jpg", inputImage);

      std::cout << "Image received!\n";

      cv::Mat outputImage;
      std::vector<::mediapipe::NormalizedLandmarkList> multiHandLandmarks = std::vector<::mediapipe::NormalizedLandmarkList>();
      absl::Status run_status = RunMPPGraph(inputImage, outputImage, multiHandLandmarks);

      // print output image with landmarks for debugging
      cv::imwrite("/Users/elifiamuthia/Desktop/output_image.jpg", outputImage);

      std::cout << "Printing landmarks ... " << std::endl;

      // Send all landmarks as a single concatenated string
      std::string allLandmarksString;
      for (auto& landmark : multiHandLandmarks) {
          allLandmarksString += landmark.DebugString() + "##LANDMARK_DELIMITER##";
      }
      std::cout << allLandmarksString << std::endl;

      size_t landmarksSize = allLandmarksString.size();
      ssize_t sizeSent = send(clientSocket, &landmarksSize, sizeof(size_t), 0);

      if (sizeSent != sizeof(size_t)) {
          std::cerr << "Error sending landmarks size" << std::endl;
          close(clientSocket);
          return -1;
      }

      ssize_t bytesSent = send(clientSocket, allLandmarksString.c_str(), landmarksSize, 0);

      if (bytesSent < 0) {
          std::cerr << "Error sending landmarks data" << std::endl;
          close(clientSocket);
          return -1;
      }
    }
    

    close(clientSocket);
  }

  close(serverSocket);
  return EXIT_SUCCESS;
}