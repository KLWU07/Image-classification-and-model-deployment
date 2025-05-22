import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
import io


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("花卉图像分类器")
        self.root.geometry("800x600")

        # 服务器URL
        self.server_url = "http://192.168.168.5:5000/predict"

        # 创建UI组件
        self.create_widgets()

    def create_widgets(self):
        # 顶部框架
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=20)

        # 选择图像按钮
        self.select_btn = tk.Button(top_frame, text="选择花卉图像", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=10)

        # 预测按钮
        self.predict_btn = tk.Button(top_frame, text="预测", command=self.predict_image, state=tk.DISABLED)
        self.predict_btn.pack(side=tk.LEFT, padx=10)

        # 图像显示区域
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(pady=20)
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        # 结果显示区域
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(pady=20)

        self.class_label = tk.Label(self.result_frame, text="花卉种类: ", font=('Arial', 14))
        self.class_label.pack()

        self.confidence_label = tk.Label(self.result_frame, text="置信度: ", font=('Arial', 14))
        self.confidence_label.pack()

        # 添加一个显示类别对应关系的文本区域
        self.class_info = tk.Text(self.root, height=6, width=50, font=('Arial', 10))
        self.class_info.pack(pady=10)
        self.class_info.insert(tk.END, "类别对应关系:\n0: daisy\n1: dandelion\n2: roses\n3: sunflower\n4: tulips")
        self.class_info.config(state=tk.DISABLED)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="选择花卉图像",
            filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))

        if file_path:
            try:
                self.image_path = file_path
                self.display_image(file_path)
                self.predict_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图像: {str(e)}")

    def display_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((400, 400))

        img_tk = ImageTk.PhotoImage(img)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

    def predict_image(self):
        if not hasattr(self, 'image_path'):
            messagebox.showwarning("警告", "请先选择图像")
            return

        try:
            with open(self.image_path, 'rb') as img_file:
                files = {'file': img_file}
                response = requests.post(self.server_url, files=files)

                if response.status_code == 200:
                    result = response.json()
                    self.show_result(result)
                else:
                    messagebox.showerror("错误", f"预测失败: {response.text}")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")

    def show_result(self, result):
        self.class_label.config(text=f"花卉种类: {result['class']} (ID: {result['class_id']})")
        self.confidence_label.config(text=f"置信度: {result['confidence']:.2%}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()