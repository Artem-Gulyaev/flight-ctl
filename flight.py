import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
# for Rectangle/Line/Ellipce/...
from kivy.graphics import *
# window control
from kivy.core.window import Window

class PointMass:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.vx = 0.
        self.vy = 0.

class MainFlightScreen(GridLayout):

    def draw_grid(self, canvas):
        with self.canvas:
            Color(1.,1.,1.)
            # https://kivy.org/doc/stable/api-kivy.graphics.html
            for i in range(self.grid_xticks):
                Line(points=[float(i) * self.size[0] / float(self.grid_xticks - 1), 0.0
                             , float(i) * self.size[0] / float(self.grid_xticks - 1), self.size[1]]
                     , width=1)
            for j in range(self.grid_yticks):
                Line(points=[0.0, float(j) * self.size[1] / float(self.grid_yticks - 1)
                             , self.size[0], float(j) * self.size[1] / float(self.grid_yticks - 1)]
                     , width=1)

    def draw_scene(self, *args):
        self.canvas.clear()

        self.draw_grid(self.canvas)

        with self.canvas:
            Color(0.0,1.0,0.0)
            Line(circle=(self.point.x, self.point.y, self.point_size))
            Line(points=[self.point.x - self.point_size / 2.
                         , self.point.y
                         , self.point.x + self.point_size / 2.
                         , self.point.y]
                 , width=1)
            Line(points=[self.point.x
                         , self.point.y - self.point_size / 2.
                         , self.point.x
                         , self.point.y + self.point_size / 2.]
                 , width=1)

    def __init__(self, **kwargs):
        super(MainFlightScreen, self).__init__(**kwargs)
        self.cols = 2
        self.add_widget(Label(text="Visualization"))

        # we will fly on a point mass now
        self.point = PointMass()
        self.grid_xscale_mpp = 1.0
        self.grid_yscale_mpp = 1.0
        self.grid_xticks = 10
        self.grid_yticks = 10

        self.point_size = 10

        self.size = Window.size

        self.draw_scene()

class MyApp(App):

    def build(self):
        return MainFlightScreen()

if __name__ == '__main__':
    MyApp().run()
