import math
import numpy as np

import kivy
kivy.require('1.11.1')

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
# for Rectangle/Line/Ellipce/...
from kivy.graphics import *
# window control
from kivy.core.window import Window
# Clock
from kivy.clock import Clock

# Simple point mass model
class PointMassModel:

    # @m mass of the point in KG
    def __init__(self):
        self.x = 10.
        self.y = 10.
        self.vx = 2.
        self.vy = 1.0
        self.t = 0.
        self.ax = 0.
        self.ay = -0.002
        self.m = 1.0

        # platform
        self.platform_x = 8.0
        self.platform_size = (20., 1.)
        self.platform_vx = 0.0
        self.platform_ax = 0.0

        # unclonable
        self.prediction = None

    # adopts all parameters from other instance
    def clone_from(self, other):
        self.x = other.x
        self.y = other.y
        self.vx = other.vx
        self.vy = other.vy
        self.t = other.t
        self.ax = other.ax
        self.ay = other.ay
        self.m = other.m
        self.platform_x = other.platform_x
        self.platform_size = other.platform_size
        self.platform_vx = other.platform_vx
        self.platform_ax = other.platform_ax

    # updates the model time
    def iterate(self, dt, update_prediction=False):
        self.t += dt
        f_total = self.total_force()
        self.ax = f_total[0]/self.m
        self.ay = f_total[1]/self.m

        p_vx = self.vx
        p_vy = self.vy

        self.vx += dt * self.ax
        self.vy += dt * self.ay

        vx_eff = (p_vx + self.vx) / 2.0
        vy_eff = (p_vy + self.vy) / 2.0

        self.x += self.ax * dt**2.0 / 2.0 + vx_eff * dt
        self.y += self.ay * dt**2.0 / 2.0 + vy_eff * dt

        #if (self.y < 0):
        #    self.y = 0
        #    self.vy = -self.vy
        if (self.y > 40):
            self.y = 40
            self.vy = -self.vy
        if (self.x < 0):
            self.x = 0
            self.vx = -self.vx
        if (self.x > 150.0):
            self.x = 150.0
            self.vx = -self.vx

        if (self.y < self.platform_size[1] and self.y >= 0
                and self.x >= self.platform_x - self.platform_size[0]
                and self.x <= self.platform_x + self.platform_size[0]):
            self.y = self.platform_size[1]
            self.vy = -1.0 * self.vy

        # platform
        self.platform_vx += dt * self.platform_ax
        self.platform_x += self.platform_ax * dt**2.0 / 2.0 + self.platform_vx * dt

        if (self.platform_x - self.platform_size[0] / 2. < 0.):
            self.platform_x = self.platform_size[0] / 2.
            self.platform_vx = -self.platform_vx

        if (self.platform_x + self.platform_size[0] / 2. > 150.):
            self.platform_x = 150. - self.platform_size[0] / 2.
            self.platform_vx = -self.platform_vx

        # do future prediction
        if update_prediction:
            self.prediction = self.predict()
        
    # RETURNS: the tuple of (F_x, F_y) force components
    def total_force(self):
        ff = self.friction_force_2d(self.vx, self.vy)
        gf = self.g_force()
        return [ c1 + c2 for c1,c2 in zip(ff,gf) ]

    # RETURNS: the tuple of (F_x, F_y) graviational force components
    def g_force(self):
        return (0.0, -0.02)

    # RETURNS: the dynamic friction force 2d vector
    #   for given parameters of flight
    #   (f_x, f_y)
    def friction_force_2d(self, vx, vy):
        V1_FRICTION_COEFF = 0.000
        V2_FRICTION_COEFF = 0.000
        V3_FRICTION_COEFF = 0.000

        v_abs = math.sqrt(vx**2. + vy**2.)
        v_dir = (vx, vy)
        F_abs = ( V1_FRICTION_COEFF
                  + V2_FRICTION_COEFF * v_abs +
                  + V3_FRICTION_COEFF * v_abs**2)
        
        F_f = (-v_dir[0] * F_abs, -v_dir[1] * F_abs)
        return F_f

    # the prediction of the model future dynamics
    # RETURNS:
    #   numpy array of points (t, x, y)
    #   NOTE: time starts from current position
    def predict(self):
        vmodel = PointMassModel()
        vmodel.clone_from(self)
        vmodel.t = 0.
        trajectory = np.zeros((10, 3))
        predict_dt = 0.1
        for i in range(trajectory.shape[0]):
            vmodel.iterate(predict_dt)
            trajectory[i] = np.array((vmodel.t, vmodel.x, vmodel.y))
        return trajectory
            
# Just draws a grid with given parameters
class Grid2d:

    # @meters_per_pixel_x the scale along X axis
    # @meters_per_pixel_y the scale along Y axis
    # @grid_geom_m (x1,y1,x2,y2)
    def __init__(self, meters_per_pixel_x=0.1
                     , meters_per_pixel_y=0.1
                     , main_x_ticks=10
                     , main_y_ticks=10
                     , grid_geom_m=(0.,0.,10.,10.)):
        self.grid_xscale_mpp = meters_per_pixel_x
        self.grid_yscale_mpp = meters_per_pixel_y
        self.grid_xticks = main_x_ticks
        self.grid_yticks = main_y_ticks
        self.grid_geom_m = grid_geom_m

    # @viewport_geom_m (x1_meters, y1_meters, x2_meters, y2_meters)
    # @viewport_size_pix (width, height)
    #
    # RETURNS:
    #   x in pixels for given x in meters for our viewport 
    def xm2xp(self, x_m, viewport_geom_m, viewport_size_pix):
        return ((x_m - viewport_geom_m[0]) * viewport_size_pix[0]
                    / (viewport_geom_m[2] - viewport_geom_m[0]))

    # @viewport_geom_m (x1_meters, y1_meters, x2_meters, y2_meters)
    # @viewport_size_pix (width, height)
    #
    # RETURNS:
    #   y in pixels for given y in meters for our viewport 
    def ym2yp(self, y_m, viewport_geom_m, viewport_size_pix):
        return ((y_m - viewport_geom_m[1]) * viewport_size_pix[1]
                    / (viewport_geom_m[3] - viewport_geom_m[1]))

    # draws the grid on a given canvas
    # @vport_geom_m (x1,y1,x2,y2) in meters
    # @vport_size_pix (w,h) of viewport in pix
    def draw(self, canvas, grid_geom_m, vport_geom_m, vport_size_pix):
        return
        with canvas:
            Color(1.,1.,1.)

        # https://kivy.org/doc/stable/api-kivy.graphics.html
        for i in range(self.grid_xticks):
            xm = grid_geom_m[0] + (grid_geom_m[2] - grid_geom_m[0]) * float(i) / float(self.grid_xticks)
            xp = self.xm2xp(xm, vport_geom_m, vport_size_pix)

            yp_min = self.ym2yp(grid_geom_m[1], vport_geom_m, vport_size_pix)
            yp_max = self.ym2yp(grid_geom_m[3], vport_geom_m, vport_size_pix)

            with canvas:
                Line(points=[xp, yp_min,xp , yp_max], width=1)

        for j in range(self.grid_yticks):
            ym = grid_geom_m[1] + (grid_geom_m[3] - grid_geom_m[1]) * float(j) / float(self.grid_yticks)
            yp = self.ym2yp(ym, vport_geom_m, vport_size_pix)

            xp_min = self.xm2xp(grid_geom_m[0], vport_geom_m, vport_size_pix)
            xp_max = self.xm2xp(grid_geom_m[2], vport_geom_m, vport_size_pix)

            with canvas:
                Line(points=[xp_min, yp, xp_max, yp], width=1)


# Visual representation of the 2d + 1t dot mass model
class ModelVisualization2d1t(Widget):

    def __init__(self, **kwargs):
        super(ModelVisualization2d1t, self).__init__(**kwargs)
        self.grid = Grid2d()
        self.vport_geom_m = (-0, 0, 150, 50)
        self.grid_geom_m = (-0, 0, 150, 50)

    def draw(self, model):
        self.canvas.clear()

        self.grid.draw(self.canvas, grid_geom_m=self.grid_geom_m
                       , vport_geom_m=self.vport_geom_m
                       , vport_size_pix=self.size)


        with self.canvas:
            Color(0.6,0.6,1.0)
            MARK_SIZE_PIX=20
            cx_p = self.grid.xm2xp(model.x, self.vport_geom_m, self.size)
            cy_p = self.grid.ym2yp(model.y, self.vport_geom_m, self.size)

            Line(circle=(cx_p, cy_p, MARK_SIZE_PIX))
            Line(points=[cx_p - MARK_SIZE_PIX / 2. , cy_p
                         , cx_p + MARK_SIZE_PIX / 2. , cy_p]
                 , width=1)
            Line(points=[cx_p, cy_p - MARK_SIZE_PIX / 2.
                         , cx_p, cy_p + MARK_SIZE_PIX / 2.]
                 , width=1)

        # draw platform
        with self.canvas:
            #      R    G    B
            xl = self.grid.xm2xp(model.platform_x - model.platform_size[0]/2
                                 , self.vport_geom_m, self.size)
            xr = self.grid.xm2xp(model.platform_x + model.platform_size[0]/2
                                 , self.vport_geom_m, self.size)
            yb = self.grid.ym2yp(0, self.vport_geom_m, self.size)
            yt = self.grid.ym2yp(model.platform_size[1], self.vport_geom_m, self.size)

            Color(1.0, 1.0, 0.0)
            Rectangle(pos=(xl, yb), size=(xr - xl, yt - yb))

        with self.canvas:
            MARK_SIZE_PIX=3
            if model.prediction is not None:
                for i in (list(range(model.prediction.shape[0]))
                          + list(range(0, model.prediction.shape[0], 100))):
                    t = model.prediction[i,0]
                    x = model.prediction[i,1]
                    y = model.prediction[i,2]
                    t_max = model.prediction[-1,0]
                    if i % 100 != 0:
                        Color(1.0 - t/t_max, 1.0 - t/t_max, 1.0)
                    else:
                        Color(1.0, 0.0, 0.0)

                    cx_p = self.grid.xm2xp(x, self.vport_geom_m, self.size)
                    cy_p = self.grid.ym2yp(y, self.vport_geom_m, self.size)

                    Line(circle=(cx_p, cy_p, MARK_SIZE_PIX))
                    Line(points=[cx_p - MARK_SIZE_PIX / 2. , cy_p
                                 , cx_p + MARK_SIZE_PIX / 2. , cy_p]
                         , width=1)
                    Line(points=[cx_p, cy_p - MARK_SIZE_PIX / 2.
                                 , cx_p, cy_p + MARK_SIZE_PIX / 2.]
                         , width=1)


class MainFlightScreen(GridLayout):

    def iterate(self, dt):
        self.model.iterate(dt*6., update_prediction=True)
        self.scene.draw(self.model)
        self.wdict["model_time"].text = "Model time: %s s" % self.model.t

    def __init__(self, **kwargs):
        super(MainFlightScreen, self).__init__(**kwargs)
        self.cols = 2
        self.wdict = {}
        self.add_widget(Label(text="Visualization"))
        self.wdict["model_time"] = Label(text="")
        self.add_widget(self.wdict["model_time"])
        self.scene = ModelVisualization2d1t(size_hint_x=None
                                            , size_hint_y=None
                                            , width=800
                                            , height=700)
        self.add_widget(self.scene)

        self.model = PointMassModel()
        
        self.scene.draw(self.model)

        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self.on_keyboard_down)
        self._keyboard.bind(on_key_up=self.on_keyboard_up)

        Clock.schedule_interval(self.iterate, 0.1)

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self.on_keyboard_down)
        self._keyboard.unbind(on_key_up=self.on_keyboard_up)
        self._keyboard = None

    def on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[0] == 1073741904:
            print("Pl. Left")
            self.model.platform_ax = -1.0
        if keycode[0] == 1073741903:
            print("Pl. Right")
            self.model.platform_ax = +1.0

    def on_keyboard_up(self, keyboard, keycode):
        if keycode[0] == 1073741904:
            self.model.platform_ax = 0.
        if keycode[0] == 1073741903:
            self.model.platform_ax = 0.

class MyApp(App):

    def build(self):
        return MainFlightScreen()

if __name__ == '__main__':
    MyApp().run()
