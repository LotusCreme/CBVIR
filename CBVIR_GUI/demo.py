import os
import re
from pathlib import Path
from PIL import ImageTk, Image
from tkinter import *
from tkinter import Scrollbar, PhotoImage, filedialog
from utils.KFE_module import *
from utils.CBIR_module import *


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets")
RESULT_PATH = OUTPUT_PATH / Path(r"results")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


class Demo():
    def __init__(self, master):
        self.size = [900, 577]
        self.frame = [None]*6
        self.cur_frame = 0
        self.window = master

        # define frame0
        self.frame[0] = Frame(self.window, width=self.size[0], height=self.size[1], bd=0)
        self.canvas_home = Canvas(
            self.frame[0],
            bg = "#94D8EE",
            width = self.size[0],
            height = self.size[1],
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
            )
        self.canvas_home.place(x=0, y=0)

        self.canvas_home.create_rectangle(
            49.99969482421875,
            141.5,
            852.0003662109375,
            143.5,
            fill="#000000",
            outline="")


        ##################################################
        ###########  define frame0's buttons  ############
        ##################################################

        self.b_img_0_0 = PhotoImage(
            file=relative_to_assets("frame0/button_1.png"))
        self.button00 = Button(
            self.frame[0],
            image=self.b_img_0_0,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_start,
            relief="flat"
        )
        self.button00.place(
            x=347.0,
            y=284.0,
            width=226.0,
            height=77.0
        )

        self.b_img_0_1 = PhotoImage(
            file=relative_to_assets("frame0/button_2.png"))
        self.button01 = Button(
            self.frame[0],
            image=self.b_img_0_1,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_help,
            relief="flat"
        )
        self.button01.place(
            x=77.0,
            y=418.0,
            width=203.0,
            height=65.0
        )

        self.b_img_0_2 = PhotoImage(
            file=relative_to_assets("frame0/button_3.png"))
        self.button02 = Button(
            self.frame[0],
            image=self.b_img_0_2,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_setting,
            relief="flat"
        )
        self.button02.place(
            x=355.0,
            y=418.0,
            width=203.0,
            height=65.0
        )

        self.b_img_0_3 = PhotoImage(
            file=relative_to_assets("frame0/button_4.png"))
        self.button03 = Button(
            self.frame[0],
            image=self.b_img_0_3,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_about,
            relief="flat"
        )
        self.button03.place(
            x=625.0,
            y=418.0,
            width=203.0,
            height=65.0
        )

        # define frame0's images
        self.image00 = PhotoImage(file=relative_to_assets("frame0/image_1.png"))
        self.canvas_home.create_image(
        452.0,
        95.0,
        image = self.image00)

        # designer tag image is self.image01
        self.image01 = PhotoImage(file=relative_to_assets("frame0/image_2.png"))
        self.canvas_home.create_image(
        828.0,
        553.0,
        image = self.image01)

        self.image02 = PhotoImage(file=relative_to_assets("frame0/image_3.png"))
        self.canvas_home.create_image(
        452.0,
        196.0,
        image = self.image02)

        self.frame[0].pack()


        ##################################################
        ############  define frame1 - about  #############
        ##################################################
        
        self.frame[1] = Frame(master, width=self.size[0], height=self.size[1], bd=0)
        self.canvas_about = Canvas(
            self.frame[1],
            bg = "#94D8EE",
            width = self.size[0],
            height = self.size[1],
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
            )
        self.canvas_about.place(x=0, y=0)

        # reusable home button is b_img_1_0
        self.b_img_1_0 = PhotoImage(
            file=relative_to_assets("frame1/button_1.png"))
        self.button10 = Button(
            self.frame[1],
            image=self.b_img_1_0,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_home,
            relief="raised"
        )
        self.button10.place(
            x=66.0,
            y=469.0,
            width=110.0,
            height=96.0
        )

        # designer tag image is self.image01
        self.canvas_about.create_image(
        828.0,
        553.0,
        image = self.image01)

        self.image11 = PhotoImage(file=relative_to_assets("frame1/image_2.png"))
        self.canvas_about.create_image(
        175.0,
        60.0,
        image = self.image11)

        self.image12 = PhotoImage(file=relative_to_assets("frame1/image_3.png"))
        self.canvas_about.create_image(
        448.0,
        294.0,
        image = self.image12)


        
        ##################################################
        #############  define frame2 - help  #############
        ##################################################

        self.frame[2] = Frame(master, width=self.size[0], height=self.size[1], bd=0)
        self.canvas_help = Canvas(
            self.frame[2],
            bg = "#94D8EE",
            width = self.size[0],
            height = self.size[1],
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
            )
        self.canvas_help.place(x=0, y=0)

        # reusable home button image is b_img_1_0
        self.button20 = Button(
            self.frame[2],
            image=self.b_img_1_0,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_home
        )
        self.button20.place(
            x=66.0,
            y=469.0,
            width=110.0,
            height=96.0
        )

        # designer tag image is self.image01
        self.canvas_help.create_image(
        828.0,
        553.0,
        image = self.image01)

        self.image21 = PhotoImage(file=relative_to_assets("frame2/image_3.png"))
        self.canvas_help.create_image(
        172.0,
        55.0,
        image = self.image21)

        self.image22 = PhotoImage(file=relative_to_assets("frame2/image_2.png"))
        self.canvas_help.create_image(
        444.0,
        288.0,
        image = self.image22)


        ##################################################
        ###########  define frame3 - setting  ############
        ##################################################
        
        self.frame[3] = Frame(master, width=self.size[0], height=self.size[1], bd=0)
        self.canvas_setting = Canvas(
            self.frame[3],
            bg = "#94D8EE",
            width = self.size[0],
            height = self.size[1],
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
            )
        self.canvas_setting.place(x=0, y=0)

        # reusable home button image is b_img_1_0
        self.button30 = Button(
            self.frame[3],
            image=self.b_img_1_0,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_home
        )
        self.button30.place(
            x=66.0,
            y=469.0,
            width=110.0,
            height=96.0
        )

        # designer tag image is self.image01
        self.canvas_setting.create_image(
        828.0,
        553.0,
        image = self.image01)

        self.image31 = PhotoImage(file=relative_to_assets("frame3/image_2.png"))
        self.canvas_setting.create_image(
        183.0,
        61.0,
        image = self.image31)

        # self.image32 = PhotoImage(file=relative_to_assets("frame3/image_3.png"))
        # self.canvas_help.create_image(
        # 440.0,
        # 295.0,
        # image = self.image32)


        ##################################################
        ############  define frame4 - start  #############
        ##################################################
        
        self.frame[4] = Frame(master, width=self.size[0], height=self.size[1], bd=0)
        self.canvas_start = Canvas(
            self.frame[4],
            bg = "#94D8EE",
            width = self.size[0],
            height = self.size[1],
            bd = 0,
            highlightthickness = 0,
            relief = "ridge"
            )
        self.canvas_start.place(x=0, y=0)
        
        self.canvas_start.create_rectangle(
            55.0,
            162.0,
            857.000732421875,
            164.0,
            fill="#000000",
            outline="")
        

        # reusable home button image is b_img_1_0
        self.button40 = Button(
            self.frame[4],
            image=self.b_img_1_0,
            borderwidth=0,
            highlightthickness=0,
            command= self.to_home
        )
        self.button40.place(
            x=71.0,
            y=477.0,
            width=86.0,
            height=82.0
        )

        self.b_img_4_1 = PhotoImage(
            file=relative_to_assets("frame4/button_2.png"))
        self.button41 = Button(
            self.frame[4],
            borderwidth=0,
            highlightthickness=0,
            command=self.on_click_run,
            image=self.b_img_4_1,
            relief="flat"
        )
        self.button41.place(
            x=757.0,
            y=28.0,
            width=76.0,
            height=108.0
        )

        self.b_img_4_2 = PhotoImage(
            file=relative_to_assets("frame4/button_3.png"))
        self.button42 = Button(
            self.frame[4],
            borderwidth=0,
            highlightthickness=0,
            command=self.get_video,
            image=self.b_img_4_2,
            relief="flat"
        )
        self.button42.place(
            x=65.0,
            y=37.0,
            width=634.0,
            height=46.0
        )

        self.b_img_4_3 = PhotoImage(
            file=relative_to_assets("frame4/button_4.png"))
        self.button43 = Button(
            self.frame[4],
            borderwidth=0,
            highlightthickness=0,
            command=self.get_image,
            image=self.b_img_4_3,
            relief="flat"
        )
        self.button43.place(
            x=65.0,
            y=93.0,
            width=634.0,
            height=46.0
        )

        self.label_error = Label(
            self.frame[4], 
            text=" Error: Invalid input. Please provide valid inputs.", 
            font=("Arial", 13), 
            highlightthickness=0, 
            bd=0,
            relief="flat",
            image=PhotoImage(file=relative_to_assets("frame4/image_2.png")),
            compound="center")

        ##################################################
        #######  define frame5 - display results  ########
        ##################################################

        self.frame[5] = Frame(master, width=self.size[0], height=self.size[1], bd=0)
        self.canvas_display = Canvas(self.frame[5], width=self.size[0]-100, height=self.size[1])
        self.canvas_display.place(x=0, y=0)

        self.scrollbar = Scrollbar(self.frame[5], orient=VERTICAL, command=self.canvas_display.yview)
        self.scrollbar.place(relx=1, rely=0, relheight=1, anchor=NE)
        self.canvas_display.configure(yscrollcommand=self.scrollbar.set)

        self.inner_frame = Frame(self.canvas_display)
        self.canvas_display.create_window((0, 0), window=self.inner_frame, anchor="nw")
        

        # reusable home button image is b_img_1_0
        # self.button50 = Button(
        #     self.frame[5],
        #     image=self.b_img_1_0,
        #     borderwidth=0,
        #     highlightthickness=0,
        #     command= self.to_home
        # )
        # self.button50.place(
        #     x=66.0,
        #     y=469.0,
        #     width=110.0,
        #     height=96.0
        # )


        ##################################################
        ##################  Defining  ####################
        ##################################################
        # designer tag image is self.image01
        self.canvas_start.create_image(
        828.0,
        553.0,
        image = self.image01)

        self.video_path = False
        self.img_path = False
        self.save_to = False

    ###############################################################
    ###############  Transition Functions  ########################
    ###############################################################

    def to_home(self):
        self.frame[self.cur_frame].pack_forget()      
        self.frame[0].pack()
        self.cur_frame = 0

    def to_about(self):
        self.frame[self.cur_frame].pack_forget() 
        self.frame[1].pack()
        self.cur_frame = 1

    def to_help(self):
        self.frame[self.cur_frame].pack_forget() 
        self.frame[2].pack()
        self.cur_frame = 2

    def to_setting(self):
        self.frame[self.cur_frame].pack_forget() 
        self.frame[3].pack()
        self.cur_frame = 3

    def to_start(self):
        self.frame[self.cur_frame].pack_forget() 
        self.frame[4].pack()
        self.cur_frame = 4

    def to_display(self):
        self.frame[self.cur_frame].pack_forget() 
        self.frame[5].pack()
        self.cur_frame = 5


    #############################################################
    #############################################################

    def get_video(self):
        print("Please select the query video.")
        self.video_path = filedialog.askopenfilename()
        print("video_path: ", self.video_path)

        if self.video_path:
            temp = self.video_path
            temp1 = re.sub(r"[\\/]+", "/", temp)
            parts = temp1.split("/")[-2:] 
            display_path = os.path.join("...", *parts)
            self.save_to = os.path.join(RESULT_PATH, ('keyframes_'+ Path(self.video_path).stem))

            label_video = Label(
                self.frame[4], 
                text=display_path, 
                font=("Arial", 13), 
                highlightthickness=0, 
                bd=0,
                relief="flat",
                image=PhotoImage(file=relative_to_assets("frame4/image_2.png")),
                compound="center")
            
            label_video.place(x=349.0, y=43.0)

        else:
            # print("..", type(self.video_path))
            label_video = Label(
                self.frame[4], 
                text="Invalid path, please upload again.", 
                font=("Arial", 13), 
                highlightthickness=0, 
                bd=0,
                relief="flat",
                image=PhotoImage(file=relative_to_assets("frame4/image_2.png")),
                compound="center")
            
            label_video.place(x=349.0, y=43.0)

    def get_image(self): 
        print("Please select the query image.")
        self.img_path = filedialog.askopenfilename()
        print("image_path: ", self.img_path)

        if self.img_path:
            temp = self.img_path
            temp1 = re.sub(r"[\\/]+", "/", temp)
            parts = temp1.split("/")[-2:] 
            display_path = os.path.join("...", *parts)

            label_image = Label(
                self.frame[4], 
                text=display_path, 
                font=("Arial", 13), 
                highlightthickness=0, 
                bd=0,
                relief="flat",
                image=PhotoImage(file=relative_to_assets("frame4/image_2.png")),
                compound="center")
            
            label_image.place(x=349.0, y=99.0)

        else:
            # print("..", type(self.img_path))
            label_image = Label(
                self.frame[4], 
                text="Invalid path, please upload again.", 
                font=("Arial", 13), 
                highlightthickness=0, 
                bd=0,
                relief="flat",
                image=PhotoImage(file=relative_to_assets("frame4/image_2.png")),
                compound="center")
            
            label_image.place(x=349.0, y=99.0)

            
    def read_display(self):
        f = open(self.results_txt) 
        # print(self.results_txt)
        idx = []
        for line in f:
            line = line.rstrip("\n")
            idx.append(line)
            # print(line)
        f.close()

        self.imgs = []
        text_labels = []
        column = 0
        row = 0
        window_width = self.size[0] - 100
        desired_width = window_width // 4  # Display 4 images in one row

        for i in range(len(idx)):
            img = Image.open(os.path.join(self.final_save_to, idx[i]))
            # print(img.format)
            width, height = img.size
            aspect_ratio = width / height
            new_height = int(desired_width / aspect_ratio)
            img = img.resize((desired_width, new_height), Image.LANCZOS)
            self.imgs.append(ImageTk.PhotoImage(img))

            label = Label(self.inner_frame, image=self.imgs[-1])
            label.grid(row=row, column=column)

            text = idx[i]
            time_pattern = r"time(\d+)_(\d+)\.jpg"
            match = re.match(time_pattern, text)
            if match:
                minutes = match.group(1)
                seconds = match.group(2).zfill(2)  # Pad seconds with leading zero
                time_format = f"{minutes}:{seconds}"
            else:
                print("Invalid filename format.")

            text_label = Label(self.inner_frame, text=time_format)
            text_label.grid(row=row + 1, column=column)
            text_labels.append(text_label)

            column += 1
            if column >= 4:
                column = 0
                row += 2


    def on_click_run(self):

        if self.video_path and self.img_path:
            
            if self.label_error.winfo_exists():
                self.label_error.place_forget()

            ############# KFE ###############
            self.label_KFE_img = PhotoImage(file=relative_to_assets("frame4/image_2.png"))
            self.label_KFE = Label(
                self.frame[4], 
                text="ʕ •ᴥ•ʔ  Extracting keyframes... ", 
                font=("Arial", 13), 
                highlightthickness=0, 
                bd=0,
                relief="flat",
                image=self.label_KFE_img,
                compound="center")

            self.label_KFE.place(x=284.0, y=200.0)
            self.window.update()

            if self.label_KFE.winfo_manager():
                self.exe_time_1, self.final_save_to = KFE(self.video_path, self.save_to)

            ############# CBIR ###############
            self.label_KFE.config(text=f" ^•ﻌ•^  Keyframe extraction is done. [{self.exe_time_1:.2f}s]")
            self.label_CBIR = Label(
                self.frame[4], 
                text="ʕ •ᴥ•ʔ  Seaching ... ", 
                font=("Arial", 13), 
                highlightthickness=0, 
                bd=0,
                relief="flat",
                image=self.label_KFE_img,
                compound="center")

            self.label_CBIR.place(x=284.0, y=260.0)
            self.window.update()

            if self.label_CBIR.winfo_manager():
                self.results_txt, self.exe_time_2 = search(self.img_path, self.final_save_to, RESULT_PATH)


            ############# FINISH #################
            self.label_CBIR.config(text=f" ^•ﻌ•^  Searching is done. [{self.exe_time_2:.2f}s]")
            self.button_finish = Button(
                self.frame[4],
                image=self.label_KFE_img,
                text=" ʕ •ᴥ•ʔ  Click here to see results.",
                font=("Arial", 13), 
                borderwidth=5,
                highlightthickness=0,
                command=self.to_display,
                relief="flat",
                compound="center"
            )
            self.button_finish.place(x=284.0, y=320.0)
            self.window.update()

            self.read_display()
            self.inner_frame.update_idletasks()
            self.canvas_display.configure(scrollregion=self.canvas_display.bbox("all"))

        else:
            self.label_error.place(x=290.0, y=289.0)




if __name__ == "__main__":
    window = Tk()
    window.geometry("900x577")
    window.title('CBVIR')
    photo = PhotoImage(file = ASSETS_PATH / Path("icon/find.png"))
    window.iconphoto(False, photo)
    app = Demo(window)
    window.mainloop()

