#include <opencv2/opencv.hpp>
#include <fstream>  
// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;
// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);
void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}
vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}
Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    // Unwrap detections.
    for (Mat output : outputs)
    {
        float data = (float)output.data;
        for (int i = 0; i < output.rows; i++, data += output.cols)
        {
            Mat scores = output.row(i).colRange(5, output.cols);
            Point class_id_point;
            double confidence;
            // Get the value and location of the maximum score.
            minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
            if (confidence > CONFIDENCE_THRESHOLD)
            {
                int center_x = (int)(data[0] * x_factor);
                int center_y = (int)(data[1] * y_factor);
                int width = (int)(data[2] * x_factor);
                int height = (int)(data[3] * y_factor);
                int left = center_x - width / 2;
                int top = center_y - height / 2;
                class_ids.push_back(class_id_point.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    // Draw bounding boxes.
    for (int i : indices)
    {
        Rect box = boxes[i];
        draw_label(input_image, class_name[class_ids[i]], box.x, box.y);
        rectangle(input_image, box, BLUE, THICKNESS);
    }
    return input_image;
}
int main(int argc, char** argv)
{
    // Load class names.
    ifstream class_file("coco.names");
    vector<string> class_name;
    string line;
    while (getline(class_file, line))
    {
        class_name.push_back(line);
    }
    // Load the network.
    Net net = readNet("yolov5s.onnx");
    // Open the default camera.
    VideoCapture cap(0);
    // Check if we succeeded.
    if (!cap.isOpened())
    {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    // Create a window.
    namedWindow("Object Detection", WINDOW_NORMAL);
    // Loop until 'q' is pressed.
    while (true)
    {
        // Capture frame-by-frame.
        Mat frame;
        cap >> frame;
        // Pre-process.
        vector<Mat> outputs = pre_process(frame, net);
        // Post-process.
        Mat output_image = post_process(frame, outputs, class_name);
        // Display the resulting frame.
        imshow("Object Detection", output_image);
        // Press  ESC on keyboard to exit.
        char c = (char)waitKey(25);
        if (c == 27)
        {
            break;
        }
    }
    // When everything done, release the video capture object.
    cap.release();
    // Closes all the frames.
    destroyAllWindows();
    return 0;
}
