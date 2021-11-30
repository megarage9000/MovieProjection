from tkinter import *
from tkinter import ttk

from AdultQuestion import ui_helper


def main_page():
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
    ttk.Label(frm, text="Ask Q").grid(column=0, row=2)
    ttk.Button(frm, text="Ask About Adult Movies", command=classifier_page).grid(column=1, row=2)
    root.mainloop()


def classifier_page():
    frm = ttk.Frame(root, padding=10)
    ttk.Label(frm, text="This command will take a while").grid(column=2, row=0)
    models = ui_helper()
    frm = ttk.Frame(Tk(className="Is the Movie an Adult Film"), padding=10)
    frm.grid()
    ttk.Label(frm, text="By Average Rating").grid(column=0, row=0)
    classifier_tab(frm, 1, models[0])
    ttk.Label(frm, text="By Number of Ratings").grid(column=0, row=6)
    classifier_tab(frm, 7, models[1])
    ttk.Label(frm, text="By Both Average Rating and Number of Ratings").grid(column=0, row=12)
    classifier_tab(frm, 13, models[2])
    pass

def classifier_tab(frm, start_row, models):
    x_valid, y_valid = models[0]
    ttk.Label(frm, text="Gauss Model: ").grid(column=0, row=start_row + 0)
    ttk.Label(frm, text=models[1].score(x_valid, y_valid)).grid(column=1, row=start_row + 0)

    ttk.Label(frm, text="K Neighbours Model: ").grid(column=0, row=start_row + 1)
    ttk.Label(frm, text=models[2].score(x_valid, y_valid)).grid(column=1, row=start_row + 1)

    ttk.Label(frm, text="Random Forest Model: ").grid(column=0, row=start_row + 2)
    ttk.Label(frm, text=models[3].score(x_valid, y_valid)).grid(column=1, row=start_row + 2)

    ttk.Label(frm, text="Perceptron Model: ").grid(column=0, row=start_row + 3)
    ttk.Label(frm, text=models[4].score(x_valid, y_valid)).grid(column=1, row=start_row + 3)

    ttk.Label(frm, text="Models Get to Vote: ").grid(column=0, row=start_row + 4)
    ttk.Label(frm, text=models[5].score(x_valid, y_valid)).grid(column=1, row=start_row + 4)


root = Tk(className="Movie Data Comparator")
main_page()
