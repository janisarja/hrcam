import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Plot:
    def __init__(self, parent_frame, title="Plot", y_label="Red Intensity", width=5, height=4):
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.line, = self.ax.plot([], [])
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 255)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        
        # Create a Tkinter canvas for the plot and pack it
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=2)

        self.x_data = []
        self.y_data = []

    def update(self, x_value, y_value):
        # Update plot data
        self.x_data.append(x_value)
        self.y_data.append(y_value)
        
        # Update plot line
        self.line.set_xdata(self.x_data)
        self.line.set_ydata(self.y_data)
        self.ax.set_xlim(max(0, x_value - 20), x_value)
        
        # Update y-axis limits dynamically
        if len(self.x_data) % 20 == 0:
            if not None in self.y_data[-50:]:
                self.ax.set_ylim(min(self.y_data[-50:]) - 1, max(self.y_data[-50:]) + 1)
        
        # Redraw the canvas
        self.canvas.draw()