package com.nanomlp.ui;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

/**
 * 核心绘图类：负责将神经网络的数值状态绘制成"频谱图"
 * 对应方案：JavaFX Canvas 绘制矩形色块
 */
public class SpectrumCanvas extends Canvas {

    // 定义画布大小
    private static final double WIDTH = 800;
    private static final double HEIGHT = 300;

    public SpectrumCanvas() {
        super(WIDTH, HEIGHT);
        // 初始化时先画个黑色背景
        drawBackground();
    }

    private void drawBackground() {
        GraphicsContext gc = getGraphicsContext2D();
        gc.setFill(Color.BLACK);
        gc.fillRect(0, 0, WIDTH, HEIGHT);
        
        // 绘制分区标题
        gc.setFill(Color.WHITE);
        gc.fillText("Input Layer (784)", 50, 20);
        gc.fillText("Hidden Layer (32 Features)", 350, 20);
        gc.fillText("Output Layer (10)", 650, 20);
    }

    /**
     * 核心刷新方法：接收推理结果，重绘画面
     * @param inputData 输入的 784 个像素值 (0.0 - 1.0)
     * @param hiddenData 隐藏层的 32 个激活值
     * @param outputProb 输出层的 10 个概率值
     */
    /**
     * 核心刷新方法：接收推理结果，重绘画面
     */
    public void updateVisuals(float[] inputData, float[] hiddenData, float[] outputProb) {
        GraphicsContext gc = getGraphicsContext2D();
        
        // 1. 清除旧画面
        drawBackground(); 

        // ---------------------------------------------------------
        // 区域一：输入层 (左侧) - 绘制 28x28 像素矩阵
        // ---------------------------------------------------------
        double startX = 50;
        double startY = 50;
        double pixelSize = 5; 
        
        for (int i = 0; i < 784; i++) {
            int row = i / 28;
            int col = i % 28;
            float val = inputData[i];
            
            // 【修复关键点】反归一化：将模型用的 [-0.42, 2.8] 还原回 [0.0, 1.0] 用于显示
            // 也就是执行: val * std + mean
            float displayVal = val * 0.3081f + 0.1307f;
            
            // 【安全钳制】防止浮点数计算误差导致超出 0.0-1.0 的范围
            displayVal = Math.max(0.0f, Math.min(1.0f, displayVal));
            
            gc.setFill(Color.gray(displayVal)); 
            gc.fillRect(startX + col * pixelSize, startY + row * pixelSize, pixelSize, pixelSize);
        }

        // ---------------------------------------------------------
        // 区域二：隐藏层 (中间) - 绘制 32 个频谱条
        // ---------------------------------------------------------
        double midX = 350;
        double barWidth = 8;
        double maxBarHeight = 150;
        
        for (int i = 0; i < hiddenData.length; i++) {
            float val = hiddenData[i]; 
            double height = Math.min(val * 30, maxBarHeight); 
            
            // 隐藏层不需要反归一化，但为了颜色好看，我们限制颜色范围
            // 如果 val 很小，颜色亮度就低；val 很大，亮度就高
            double brightness = Math.min(1.0, 0.3 + Math.abs(val) * 0.2);
            
            gc.setFill(Color.hsb(180, 0.8, brightness));
            gc.fillRect(midX + i * (barWidth + 2), 200 - height, barWidth, height);
        }

        // ---------------------------------------------------------
        // 区域三：输出层 (右侧) - 绘制 10 个分类概率
        // ---------------------------------------------------------
        double rightX = 650;
        double probBarHeight = 15;
        
        for (int i = 0; i < outputProb.length; i++) {
            float prob = outputProb[i];
            // 概率本来就是 0-1，不需要转换，但为了安全也可以钳制一下
            double width = Math.max(0, prob * 100); 
            
            if (prob > 0.5) gc.setFill(Color.RED);
            else gc.setFill(Color.LIMEGREEN);
            
            gc.fillRect(rightX, 50 + i * (probBarHeight + 5), width, probBarHeight);
            gc.setFill(Color.WHITE);
            gc.fillText("Digit " + i, rightX - 40, 50 + i * (probBarHeight + 5) + 12);
        }
    }
}