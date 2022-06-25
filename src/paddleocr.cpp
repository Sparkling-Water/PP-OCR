#include "paddleocr.h"


// 构造函数
PPOCR::PPOCR(string cfgfile) 
{
    // 加载配置文件
    YAML::Node ocr_conf = YAML::LoadFile(cfgfile);
    // 通用参数
    m_use_gpu = ocr_conf["common"]["use_gpu"].as<bool>();
    m_use_mkldnn = ocr_conf["common"]["use_mkldnn"].as<bool>();
    m_use_tensorrt = ocr_conf["common"]["use_tensorrt"].as<bool>();
    m_gpu_id = ocr_conf["common"]["gpu_id"].as<int>();
    m_gpu_mem = ocr_conf["common"]["gpu_mem"].as<int>();
    m_cpu_num_threads = ocr_conf["common"]["cpu_num_threads"].as<int>();
    m_precision = ocr_conf["common"]["precision"].as<std::string>();
    // 前向预测参数
    m_use_det = ocr_conf["forward"]["use_det"].as<bool>();
    m_use_rec = ocr_conf["forward"]["use_rec"].as<bool>();
    m_use_cls = ocr_conf["forward"]["use_cls"].as<bool>();
    // 文字检测参数
    m_det_model = ocr_conf["det"]["det_model"].as<std::string>();
    m_max_side = ocr_conf["det"]["max_side"].as<int>();
    m_det_thresh = ocr_conf["det"]["det_thresh"].as<float>();
    m_det_box_thresh = ocr_conf["det"]["det_box_thresh"].as<float>();
    m_det_unclip_ratio = ocr_conf["det"]["det_unclip_ratio"].as<float>();
    m_det_score_mode = ocr_conf["det"]["det_score_mode"].as<std::string>();
    m_use_dilation = ocr_conf["det"]["use_dilation"].as<bool>();
    // 文字方向分类参数
    m_cls_model = ocr_conf["cls"]["cls_model"].as<std::string>();
    m_cls_thresh = ocr_conf["cls"]["cls_thresh"].as<float>();
    m_cls_batch_num = ocr_conf["cls"]["cls_batch_num"].as<int>();
    // 文字识别参数
    m_rec_model = ocr_conf["rec"]["rec_model"].as<std::string>();
    m_char_dict_path = ocr_conf["rec"]["char_dict_path"].as<std::string>();
    m_rec_batch_num = ocr_conf["rec"]["rec_batch_num"].as<int>();
    m_rec_img_h = ocr_conf["rec"]["rec_img_h"].as<int>();
    m_rec_img_w = ocr_conf["rec"]["rec_img_w"].as<int>();

    // 创建推理对象
    if (m_use_det) 
    {
        m_detector_ = new DBDetector(
            m_det_model, m_use_gpu, m_gpu_id, m_gpu_mem,
            m_cpu_num_threads, m_use_mkldnn, m_max_side,
            m_det_thresh, m_det_box_thresh, m_det_unclip_ratio,
            m_det_score_mode, m_use_dilation, m_use_tensorrt,
            m_precision);
    }
    if (m_use_cls) 
    {
        m_classifier_ = new Classifier(
            m_cls_model, m_use_gpu, m_gpu_id, m_gpu_mem,
            m_cpu_num_threads, m_use_mkldnn, m_cls_thresh,
            m_use_tensorrt, m_precision, m_cls_batch_num);
    }
    if (m_use_rec) 
    {
        m_recognizer_ = new CRNNRecognizer(
            m_rec_model, m_use_gpu, m_gpu_id, m_gpu_mem,
            m_cpu_num_threads, m_use_mkldnn, m_char_dict_path,
            m_use_tensorrt, m_precision, m_rec_batch_num,
            m_rec_img_h, m_rec_img_w);
    }
};


// 析构函数
PPOCR::~PPOCR() 
{
    if (m_detector_ != nullptr) 
    {
        delete m_detector_;
    }
    if (m_classifier_ != nullptr) 
    {
        delete m_classifier_;
    }
    if (m_recognizer_ != nullptr) 
    {
        delete m_recognizer_;
    }
};


// 文字检测
void PPOCR::OCRDet(cv::Mat img, std::vector<OCRPredictResult> &ocr_results) 
{
    std::vector<std::vector<std::vector<int>>> boxes;
    std::vector<double> det_times;
    // 执行检测
    m_detector_->Run(img, boxes, det_times);
    // 结果保存
    for (int i = 0; i < boxes.size(); ++i) 
    {
        OCRPredictResult res;
        res.box = boxes[i];
        ocr_results.push_back(res);
    }
}


// 文字方向分类
void PPOCR::OCRCls(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results)
{
    std::vector<int> cls_labels(img_list.size(), 0);
    std::vector<float> cls_scores(img_list.size(), 0);
    std::vector<double> cls_times;
    // 执行分类
    m_classifier_->Run(img_list, cls_labels, cls_scores, cls_times);
    // 结果保存
    for (int i = 0; i < cls_labels.size(); ++i) 
    {
        ocr_results[i].cls_label = cls_labels[i];
        ocr_results[i].cls_score = cls_scores[i];
    }
}


// 文字识别
void PPOCR::OCRRec(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results)
{
    std::vector<std::string> rec_texts(img_list.size(), "");
    std::vector<float> rec_text_scores(img_list.size(), 0);
    std::vector<double> rec_times;
    // 执行识别
    m_recognizer_->Run(img_list, rec_texts, rec_text_scores, rec_times);
    // 结果保存
    for (int i = 0; i < rec_texts.size(); ++i)
    {
        ocr_results[i].text = rec_texts[i];
        ocr_results[i].score = rec_text_scores[i];
    }
}


// 主流程
void PPOCR::OCR(cv::Mat srcImg, std::vector<std::string> &ocr_res)
{
    std::vector<OCRPredictResult> ocr_results;
    // 检测
    if(m_use_det)
    {
        // 文字检测
        OCRDet(srcImg, ocr_results);
        // 图像文字区域裁剪
        std::vector<cv::Mat> img_list;
        for (int i = 0; i < ocr_results.size(); ++i)
        {
            cv::Mat crop_img;
            crop_img = Utility::GetRotateCropImage(srcImg, ocr_results[i].box);
            img_list.push_back(crop_img);
        }
        // 文本方向分类
        if (m_use_cls && m_classifier_ != nullptr) 
        {
            OCRCls(img_list, ocr_results);
            for (int i = 0; i < img_list.size(); ++i) 
            {
                if (ocr_results[i].cls_label % 2 == 1 && ocr_results[i].cls_score > m_cls_thresh) 
                {
                    cv::rotate(img_list[i], img_list[i], 1);
                }
            }
        }
        // 文字识别
        if (m_use_rec) 
        {
            OCRRec(img_list, ocr_results);
        }
    }
    // 不检测
    else
    {
        // 一个结果
        OCRPredictResult res;
        ocr_results.push_back(res);
        std::vector<cv::Mat> img_list;
        img_list.push_back(srcImg);
        // 文本方向分类
        if (m_use_cls && m_classifier_ != nullptr) 
        {
            OCRCls(img_list, ocr_results);
            for (int i = 0; i < img_list.size(); ++i) 
            {
                if (ocr_results[i].cls_label % 2 == 1 && ocr_results[i].cls_score > m_cls_thresh) 
                {
                    cv::rotate(img_list[i], img_list[i], 1);
                }
            }
        }
        // 文字识别
        if (m_use_rec)
        {
            OCRRec(img_list, ocr_results);
        }
    }
    
    // 提取识别结果
    for (int i = 0; i < ocr_results.size(); ++i)
        ocr_res.push_back(ocr_results[i].text);
}