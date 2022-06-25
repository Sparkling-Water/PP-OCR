#pragma once

#include "yaml-cpp/yaml.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>
#include <cstring>
#include <fstream>
#include <numeric>

#include <ocr_cls.h>
#include <ocr_det.h>
#include <ocr_rec.h>
#include <preprocess_op.h>
#include <utility.h>

using namespace paddle_infer;
using namespace PaddleOCR;

// PaddleOCR文字检测识别类
class PPOCR 
{
public:
    // 构造、析构函数
    explicit PPOCR(string cfgfile);
    ~PPOCR();
    // 主流程
    void OCR(cv::Mat srcImg, std::vector<std::string> &ocr_res);

private:
    // 文字检测对象
    DBDetector *m_detector_ = nullptr;
    // 文字方向分类对象
    Classifier *m_classifier_ = nullptr;
    // 文字识别对象
    CRNNRecognizer *m_recognizer_ = nullptr;
    // 通用参数
    bool m_use_gpu;             // 是否使用GPU
    bool m_use_mkldnn;          // 是否使用mkldnn库
    bool m_use_tensorrt;        // 是否使用tensorrt加速
    int m_gpu_id;               // GPU id
    int m_gpu_mem;              // GPU申请的内存
    int m_cpu_num_threads;      // CPU预测时的线程数，在机器核数充足的情况下，该值越大，预测速度越快
    std::string m_precision;    // 模型参数精度
    // 前向预测参数
    bool m_use_det;                 // 是否检测文字
    bool m_use_rec;                 // 是否识别文字
    bool m_use_cls;                 // 是否识别文字方向
    // 文字检测参数
    std::string m_det_model;        // 模型路径
    int m_max_side;                 // 输入图像长宽大于m_max_side时，等比例缩放图像，使得图像最长边为m_max_side
    float m_det_thresh;             // 用于过滤DB预测的二值化图像，设置为0.-0.3对结果影响不明显
    float m_det_box_thresh;         // DB后处理过滤box的阈值，如果检测存在漏框情况，可酌情减小
    float m_det_unclip_ratio;       // 表示文本框的紧致程度，越小则文本框更靠近文本
    std::string m_det_score_mode;   // slow:使用多边形框计算bbox score，fast:使用矩形框计算
    bool m_use_dilation;            // Whether use the dilation on output map
    // 文字方向分类参数
    std::string m_cls_model;        // 模型路径
    float m_cls_thresh;             // 方向分类器的得分阈值
    int m_cls_batch_num;            // 方向分类器batchsize
    // 文字识别参数
    std::string m_rec_model;        // 模型路径
    std::string m_char_dict_path;   // 字典文件路径
    int m_rec_batch_num;            // 识别模型batchsize
    int m_rec_img_h;                // 识别模型输入图像高度
    int m_rec_img_w;                // 识别模型输入图像宽度

private:
    // 文字检测
    void OCRDet(cv::Mat img, std::vector<OCRPredictResult> &ocr_results);
    // 文字识别
    void OCRRec(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results);
    // 文字方向分类
    void OCRCls(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results);
};
