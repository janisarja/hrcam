import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Plot:
    def __init__(self, parent_frame, title="Plot", y_label="Average Red Intensity", width=5, height=4):
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.line, = self.ax.plot([], [])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 255)
        self.ax.set_xlabel('Frame Count')
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        
        # Create a Tkinter canvas for the plot and pack it
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=2)

        self.x_data = []
        self.y_data = []
        self.frame_count = 0

    def update(self, new_value):
        # Update plot data
        self.frame_count += 1
        self.x_data.append(self.frame_count)
        self.y_data.append(new_value)
        
        # Update plot line
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.set_xlim(max(0, self.frame_count - 100), self.frame_count)
        
        # Update y-axis limits dynamically
        if len(self.x_data) % 20 == 0:
            if not None in self.y_data[-100:]:
                self.ax.set_ylim(min(self.y_data[-100:]) - 2, max(self.y_data[-100:]) + 2)
        
        # Redraw the canvas
        self.canvas.draw()