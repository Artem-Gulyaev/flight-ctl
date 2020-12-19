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
        self.x = 2.
        self.y = 2.
        self.vx = 8.
        self.vy = 4.0
        self.t = 0.
        self.ax = 0.
        self.ay = -0.002
        self.jet_fx = 0.
        self.jet_fy = 0.
        self.m = 1.0

        # unclonable
        self.prediction = None

    # sets the current acceleration of the MassPoint
    # NOTE:
    #   usually called by accelerator systems
    def set_acceleration(a):
        self.ax = a[0]
        self.ay = a[1]

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
        
        self.jet_fx = other.jet_fx
        self.jet_fy = other.jet_fy

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

        # do future prediction
        if update_prediction:
            self.prediction = self.predict()
        
    # RETURNS: the tuple of (F_x, F_y) force components
    def total_force(self):
        ff = self.friction_force_2d(self.vx, self.vy)
        gf = self.g_force()
        jetf = (self.jet_fx, self.jet_fy)
        return [ c1 + c2 + c3 for c1,c2,c3 in zip(ff,gf,jetf) ]

    # RETURNS: the tuple of (F_x, F_y) graviational force components
    def g_force(self):
        return (0.0, -0.02)

    # RETURNS: the dynamic friction force 2d vector
    #   for given parameters of flight
    #   (f_x, f_y)
    def friction_force_2d(self, vx, vy):
        V1_FRICTION_COEFF = 0.002
        V2_FRICTION_COEFF = 0.004
        V3_FRICTION_COEFF = 0.006

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

        # NOTE: prediction is made on the current motion state
        #    with jets immediately switched off
        vmodel.jet_fx = 0.
        vmodel.jet_fy = 0.

        trajectory = np.zeros((1000, 3))
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
        self.vport_geom_m = (-0, 0, 100, 100)
        self.grid_geom_m = (-0, 0, 100, 100)

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

        def draw_jet(main_angle):
            with self.canvas:
                Color(1.0,0.2,0.0)
                MAX_JETS = 20
                for i in range(0, MAX_JETS):
                    # in meters
                    A = 7.0
                    # in rads
                    SPREAD_ANGLE = math.pi/10.0

                    angle = main_angle + (float(i) - (MAX_JETS / 2))/10.0 * SPREAD_ANGLE
                    flame_x = model.x + A * math.cos(angle)
                    flame_y = model.y + A * math.sin(angle)

                    cx_p = self.grid.xm2xp(model.x, self.vport_geom_m, self.size)
                    cy_p = self.grid.ym2yp(model.y, self.vport_geom_m, self.size)

                    f_x = self.grid.xm2xp(flame_x, self.vport_geom_m, self.size)
                    f_y = self.grid.ym2yp(flame_y, self.vport_geom_m, self.size)
                    Line(points=[cx_p, cy_p, f_x, f_y], width=1)

        if (model.jet_fx > 0.0):
            draw_jet(math.pi)
        elif (model.jet_fx < 0.0):
            draw_jet(0.0)
        if (model.jet_fy > 0.0):
            draw_jet(3.0 * math.pi / 2.0)
        elif (model.jet_fy < 0.0):
            draw_jet(1.0 * math.pi / 2.0)

# defines a quite trivial rocket engine for 2d
# which can be controlled by the flight controlling
# class to control the flight path
class engine2d:
    
    # @gimble_angle in radians a gimble angle w.r.t.
    #    to the "rocket body" (in case of 2d material point
    #    it will be most likely just w.r.t. rocket velocity vector)
    def __init__(self):
        self.gimble_angle_rad = 0
        self.max_gimble_angular_speed = 0

# Just a fancy funny flight control class
# which aim is to get the 2d+1 mass point into the given
# point with given velocity
class FlightControl2d:
    pass


class MainFlightScreen(GridLayout):

    def iterate(self, dt):
        self.model.iterate(dt, update_prediction=True)
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
        JET_THRUST_F = 0.1
        print("@@@@ KEY code: " + str(keycode))
        if keycode[0] == 1073741904:
            print("Pl. Left")
            self.model.jet_fx = -JET_THRUST_F
        if keycode[0] == 1073741903:
            print("Pl. Right")
            self.model.jet_fx = +JET_THRUST_F
        if keycode[0] == 1073741906:
            print("Pl. Up")
            self.model.jet_fy = +JET_THRUST_F
        if keycode[0] == 1073741905:
            print("Pl. Down")
            self.model.jet_fy = -JET_THRUST_F

    def on_keyboard_up(self, keyboard, keycode):
        if keycode[0] == 1073741904:
            self.model.jet_fx = 0.
        if keycode[0] == 1073741903:
            self.model.jet_fx = 0.
        if keycode[0] == 1073741906:
            self.model.jet_fy = 0.
        if keycode[0] == 1073741905:
            self.model.jet_fy = 0.

class MyApp(App):

    def build(self):
        return MainFlightScreen()

if __name__ == '__main__':
    MyApp().run()
