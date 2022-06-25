#include <iostream>
#include "paddleocr.h"

int main(void)
{
    PPOCR ocr = PPOCR("../config/ppocr.yaml");
    
    std::vector<cv::String> cv_all_img_names;
    cv::glob("../images", cv_all_img_names);

    for (int i = 0; i < cv_all_img_names.size(); ++i)
    {
        cv::Mat srcImg = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
        std::vector<std::string> res;

        auto start = std::chrono::system_clock::now();
        ocr.OCR(srcImg, res);
        auto end = std::chrono::system_clock::now();
        std::cout << "------------------------------" << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        std::string name;
        for (int j = 0; j < res.size(); ++j)
            name += res[j];
        std::cout << name << std::endl;
    }
}