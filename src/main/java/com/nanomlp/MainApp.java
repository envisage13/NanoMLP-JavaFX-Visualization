package com.nanomlp;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import com.nanomlp.ui.SpectrumCanvas;
import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelReader;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Text;
import javafx.stage.Stage;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

public class MainApp extends Application {

    private OrtSession session;
    private OrtEnvironment env;
    private SpectrumCanvas spectrumCanvas;
    private ImageView originalImageView; // 用于显示原始 PNG
    private Text statusText; // 显示当前识别结果

    // 这里列出你刚才放入 resources/images 下的文件名
    // 你可以根据实际生成的文件名修改这里
    private final String[] imageFiles = {
        "0_7.png", "1_2.png", "2_1.png", "3_0.png", "4_4.png", 
        "5_1.png", "6_4.png", "7_9.png", "8_5.png", "9_9.png"
    };
    private int currentImageIndex = 0;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        try {
            // 1. 初始化模型
            env = OrtEnvironment.getEnvironment();
            // 注意：这里确保路径和文件名与你实际存放的一致
            InputStream modelStream = getClass().getResourceAsStream("/models/nano_mlp.onnx");
            
            if (modelStream == null) {
                System.err.println("严重错误：找不到模型文件！请检查 src/main/resources/models/nano_mlp.onnx 是否存在。");
                // 如果找不到模型，直接退出或抛出更明显的异常，防止后面空指针
                return;
            }
            
            byte[] modelArray = new byte[modelStream.available()];
            modelStream.read(modelArray);
            session = env.createSession(modelArray);

            // 2. 构建 UI 布局
            BorderPane root = new BorderPane();
            root.setPadding(new Insets(10));

            // --- 顶部控制栏 ---
            Button btnNext = new Button("下一张图片 (Next Image)");
            btnNext.setOnAction(e -> loadNextImage()); // Lambda 表达式
            
            statusText = new Text("准备就绪");
            statusText.setStyle("-fx-font-size: 16px; -fx-font-weight: bold;");

            HBox controls = new HBox(15, btnNext, statusText);
            controls.setAlignment(Pos.CENTER_LEFT);
            controls.setPadding(new Insets(0, 0, 10, 0));
            root.setTop(controls);

            // --- 中间可视化区域 ---
            spectrumCanvas = new SpectrumCanvas();
            root.setCenter(spectrumCanvas);

            // --- 左侧原始图片展示 ---
            originalImageView = new ImageView();
            originalImageView.setFitWidth(100); 
            originalImageView.setFitHeight(100);
            originalImageView.setSmooth(false); 
            
            VBox leftBox = new VBox(10, new Text("原始输入:"), originalImageView);
            leftBox.setPadding(new Insets(10));
            leftBox.setStyle("-fx-border-color: #ccc; -fx-border-width: 1px;");
            root.setLeft(leftBox);

            // 3. 启动窗口 (这里最容易出错，请检查这一行)
            Scene scene = new Scene(root, 1000, 400); 
            
            primaryStage.setTitle("NanoMLP 可视化 - 手写数字识别");
            primaryStage.setScene(scene);
            primaryStage.show();

            // 初始加载第一张图
            loadNextImage();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void loadNextImage() {
        try {
            // 循环获取图片文件名
            String fileName = imageFiles[currentImageIndex];
            currentImageIndex = (currentImageIndex + 1) % imageFiles.length;

            // 1. 加载图片文件
            InputStream is = getClass().getResourceAsStream("/images/" + fileName);
            if (is == null) {
                statusText.setText("错误：找不到图片 " + fileName);
                return;
            }
            Image image = new Image(is);
            originalImageView.setImage(image); // 更新左侧小图

            // 2. 预处理：Image -> Float Array
            // 必须严格匹配训练时的预处理：ToTensor() + Normalize((0.1307,), (0.3081,))
            float[] inputData = preprocessImage(image);

            // 3. 执行推理
            runInference(inputData, fileName);

        } catch (Exception e) {
            e.printStackTrace();
            statusText.setText("推理出错: " + e.getMessage());
        }
    }

    /**
     * 关键步骤：将 JavaFX Image 转换为模型需要的归一化 float 数组
     */
    private float[] preprocessImage(Image image) {
        float[] data = new float[784];
        PixelReader reader = image.getPixelReader();
        int width = (int) image.getWidth();
        int height = (int) image.getHeight();

        int index = 0;
        // 遍历 28x28 像素
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                // 如果图片尺寸不对，这里要做缩放处理，但我们生成的就是28x28，直接读即可
                // 读取颜色 (灰度图 r=g=b)
                Color color = reader.getColor(x, y);
                
                // 1. 获取亮度 (0.0 - 1.0)
                double brightness = color.getBrightness();
                
                // 2. 标准化 (Normalization)
                // 训练代码：transforms.Normalize((0.1307,), (0.3081,))
                // 公式：(input - mean) / std
                float normalized = (float) ((brightness - 0.1307) / 0.3081);
                
                data[index++] = normalized;
            }
        }
        return data;
    }

    private void runInference(float[] inputData, String fileName) throws Exception {
        // A. 准备 Tensor [1, 1, 28, 28]
        long[] shape = new long[]{1, 1, 28, 28};
        FloatBuffer buffer = FloatBuffer.wrap(inputData);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, buffer, shape);

        // B. 运行模型
        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input", inputTensor);
        Result result = session.run(inputs);

        // C. 解析结果
        float[][] probMatrix = (float[][]) result.get("prob").get().getValue();
        float[][] hiddenMatrix = (float[][]) result.get("hidden").get().getValue();
        
        float[] prob = probMatrix[0];
        float[] hidden = hiddenMatrix[0];

        // D. 找出概率最大的数字
        int predictedDigit = -1;
        float maxProb = -1;
        for (int i = 0; i < prob.length; i++) {
            if (prob[i] > maxProb) {
                maxProb = prob[i];
                predictedDigit = i;
            }
        }

        // E. 更新 UI
        spectrumCanvas.updateVisuals(inputData, hidden, prob); // 绘制频谱
        
        // 更新文字状态
        statusText.setText(String.format("图片: %s | AI预测: %d (置信度: %.1f%%)", 
                fileName, predictedDigit, maxProb * 100));
        
        // 记得关闭 tensor 释放内存
        inputTensor.close();
        result.close();
    }
}