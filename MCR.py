from math import *
import shapely as sh
import graphviz as gv
import random as rand
import matplotlib.pyplot as plt
import re


class Polygon:

    '''
    Simple (non-self-intersecting) polygons. This is a bare wrapper around the
    actual representation as a list of (x, y) tuples in CCW order. Do note that
    unlike some polygon formats (e.g. SVG) the start/end point is *not*
    duplicated.
    Please try to use Polygon objects as immutable objects: all functions return
    new Polygons.
    '''

    def __init__(self, points):
        self.points = points

    # Inspection Methods

    def bounds(self):
        '''
        Returns the [(x_min, y_min), (x_max, y_max)] bounding box of a set of
        points (probably a shape)
        '''
        xs, ys = zip(*self.points)
        return [(min(xs), min(ys)), (max(xs), max(ys))]

    def is_convex(self):
        # TODO
        for pt in self.points:
            if not (__on_left_hand__(pt, (self.points[:2]))
                    and __on_left_hand__(self.points[0],
                                         (self.points[-1], pt))):
                return False
        return True

    # Transform Methods
    # These follow the same form as SVG:
    # https://www.w3.org/TR/SVG/coords.html

    def matrix(self, a, b, c, d, e, f):
        '''
        Transforms the shape by multiplying matrix [ [a c e] [b d f] [0 0 1] ]
        by [ [x_1...x_n] [y_1...y_n] 1] and returning the top two (x and y)
        rows.
        '''
        return [(a*x + c*y + e, b*x + d*y + f) for (x, y) in self.points]

    def translate(self, t_x, t_y=0.0):
        '''
        Translates a set of points (probably a shape) by vector (t_x, t_y).
        '''
        t_x, t_y = vector
        return [(x + t_x, y + t_y) for (x, y) in self.points]

    def scale(self, s_x, s_y=None):
        '''
        Rescales a set of points (probably a shape) by s_x and s_y amount.
        If s_y is not provided, it is assumed to be the same as s_x.
        '''
        s_y = s_y if s_y else s_x
        return [(s_x * x, s_y * y) for (x, y) in self.points]

    def rotate(self, angle, c_x=0, c_y=0):
        '''
        Rotates a set of points (probably a shape) around (0,0) by angle.
        If c_x and c_y are provided, rotate around (c_x, c_y)
        '''
        s = sin(angle)
        c = cos(angle)

        return [(c*(x + c_x) - s*(y + c_y) - c_x,
                 c*(y + c_y) + s*(x + c_x) - c_y) for (x, y) in self.points]

    def skew_x(self, a):
        '''
        Skews the polygon along the x-axis by a radians
        '''
        return [(x + tan(a)*y, y) for (x, y) in self.points]

    def skew_y(self, skew):
        '''
        Skews the polygon along the y-axis by a radians
        '''
        return [(x, tan(a)*x + y) for (x, y) in self.points]

    # Set-Theoretic Methods

    # Helper methods

    def __on_left_hand__(point, edge):
        '''
        Is point on the left-hand side of the ray edge[0] - edge[1]?
        This includes being on the line itself
        '''
        px, py = point
        x1, y1 = edge[0]
        x2, y2 = edge[1]

        # special case for vertical lines (no slope)
        if (x1 == x2):
            return (px < x1) == (y1 < y2)

        # find the slope and y-intersect, then see if point is above or below the
        # ray
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        dist = py - (m * px + b)

        # if it's above and the ray is rightward, or below and the ray is leftward,
        # it's on the left-hand side
        return (dist >= 0) == (x1 < x2)


class MCR:

    '''
    Minimum Cover Removal. Contains a number of helper methods for investigating
    the MCR problem as described in Erickson and LaValle 2013 and Hauser 2012
    '''

    display_opts = {'alpha': 0.25, 'edgecolor': 'black', 'facecolor': 'gray'}

    def __init__(self, svg=None):
        '''
        Create an empty square to add shapes to.
        '''
        self.obstacles = []
        self.graph = []
        self.start = (0.05, 0.05)
        self.goal = (0.95, 0.95)
        # self.field = plt.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

        if svg:
            # initialize from svg file
            obstacles = __shapes_from_SVG__(self, svg)
            for o in obstacles:
                self.add_obstacle(o)

    def add_obstacle(self, shape, at=(0, 0)):
        self.obstacles.append(shape.translate(*at))

        # TODO: clip translated shape to field bounds
        #   - is this necessary?
        # TODO: add shape and overlaps to graph as well
        #   - or should this be done in two separate steps?

    def remove_obstacle(self, label):
        '''
        Removes the obstacle labeled 'label'.
        Removes the labeled vertices from the graph and contracts those edges
        '''
        # TODO: everything

        # Displaying the MCR
        pass

    def show_obstacles(self):
        '''
        Outputs the square
        '''
        # add start and goal
        plt.scatter(*zip(self.start, self.goal), color='red')

        # add obstacles
        for obstacle in self.obstacles:
            poly = plt.Polygon(obstacle, **self.display_opts)
            plt.gca().add_patch(poly)

        plt.axis()  # set to [0,1]
        plt.show()

    def show_graph(self):
        '''
        Outputs the square
        '''
        gv.Source(self.graph)

    def svg(self):
        '''
        Draws an SVG
        '''
        # TODO: I believe all I'll need is to change the output method of
        # matplotlib, then redraw.
        # Do I need an output file?
        pass

    # Helper methods

    def __shapes_from_SVG__(self, svg_file):
        try:
            f = open(svg_file)
            svg = f.readlines()
            f.close()
        except FileNotFoundError:
            print('File {} not found'.format(svg_file))

        # Now start parsing

        shapes = []
        scaled_shapes = []

        #  Viewbox - this will be scaled to 1, 1 eventually
        try:
            viewbox = [float(x)
                       for x in re.findall('viewBox="(.*?)"', s)[0].split()]
            vb_w = viewbox[2] - viewbox[0]
            vb_h = viewbox[3] - viewbox[1]
        except:
            print('Can\'t find a viewbox!')

        # Rectangles
        rects = re.findall('<rect.*?\/>', s)
        for r in rects:
            x_ = re.findall('x="([\d.]+)"', r)
            x = float(x_[0]) if x_ else 0.0

            y_ = re.findall('y="([\d.]+)"', r)
            y = float(y_[0]) if y_ else 0.0

            h_ = re.findall('height="([\d.]+)"', r)
            h = float(h_[0])

            w_ = re.findall('width="([\d.]+)"', r)
            w = float(w_[0])

            rect = sh.Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

            # Transforms
            transform = re.search('transform="(.*?)"', r)
            if transform:
                t_list = re.split(',\w*', transform)
                for t in t_list:
                    if t.startswith('matrix'):
                        vals = re.findall('matrix\w*\((.*?)\)', t)[0]
                        rect = sh.affinity.affine_transform(
                            rect, *[float(x) for x in re.split(',\w*', vals)]
                        )
                    elif t.startswith('translate'):
                        vals = re.findall('translate\w*\((.*?)\)', t)[0]
                        rect = sh.affinity.translate(
                            rect, *[float(x) for x in re.split(',\w*', vals)]
                        )
                    elif t.startswith('scale'):
                        vals = re.findall('scale\w*\((.*?)\)', t)[0]
                        rect = sh.affinity.scale(
                            rect,
                            *[float(x) for x in re.split(',\w*', vals)],
                            origin=(0, 0)
                        )
                    elif t.startswith('rotate'):
                        val = re.findall('scale\w*\((.*?)\)', t)[0]
                        rect = sh.affinity.rotate(
                            rect, float(val), origin=(0, 0), use_radians=True
                        )
                    elif t.startswith('skewX'):
                        val = re.findall('scale\w*\((.*?)\)', t)[0]
                        rect = rect.skew_x(float(vals))
                    elif t.startswith('skewY'):
                        val = re.findall('scale\w*\((.*?)\)', t)[0]
                        rect = rect.skew_y(float(vals))

            shapes.append(rect)

        # Polygons
        polygons = re.findall('<polygon.*?\/>', s)
        for p in polygons:
            polygon = []
            point_list = re.findall('points="(.*?)"', p)
            points = ([float(x) for x in point_list[0].split()])

            points.reverse()  # so that I can pop from the front...
            while points:
                polygon.append((points.pop(), points.pop()))
            polygon.pop()  # remove the doubled last point

            shapes.append(sh.Polygon(polygon))

        # N.b.: the svg viewbox starts at the top left
        # rescale to [1,1]
        for shape in shapes:
            scaled_shapes.append(
                [(x / vb_w, 1 - y / vb_h) for (x, y) in shape])

        return scaled_shapes

# Solving the thing


def solve_mcr(mcr):
    '''
    Solves the thing.
    '''
    pass


def random_mcr(obstacles=10):
    '''
    Creates an MCR with random polygonal obstacles
    '''
    mcr = MCR()
    for _ in range(obstacles):
        x = rand.random()
        y = rand.random()
        mcr.add_obstacle(random_shape(), at=(x, y))
        return mcr


def random_shape(max_sides=6, scale_factor=0.5):
    '''
    Creates a random convex polygon with vertices between [0,1] x [0,1]
    scaled to approx. [0, scale] in either direction
    '''

    shape = approx_ngon(rand.randrange(3, max_sides + 1))

    # rotate it randomly
    shape = rotate(shape, by=rand.uniform(0, 2*pi))

    # rescale to fit into about scale part of field
    b = bounds(shape)
    dims = (b[1][0] - b[0][0], b[1][1] - b[0][1])
    max_dim = max(dims)
    shape = scale(shape, scale_factor / max_dim)

    # and set the lower left of it's bounding box to [0,0]
    b = bounds(shape)
    x, y = [min(dim) for dim in zip(*b)]  # minimum of the b.b.
    shape = translate(shape, (-x, -y))

    return shape


def on_left_hand(point, edge):
    '''
    Is point on the left-hand side of the ray edge[0] - edge[1]?
    This includes being on the line itself
    '''
    px, py = point
    x1, y1 = edge[0]
    x2, y2 = edge[1]

    # special case for vertical lines (no slope)
    if (x1 == x2):
        return (px < x1) == (y1 < y2)

    # find the slope and y-intersect, then see if point is above or below the
    # ray
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    dist = py - (m * px + b)

    # if it's above and the ray is rightward, or below and the ray is leftward,
    # it's on the left-hand side
    return (dist >= 0) == (x1 < x2)


def approx_ngon(n, variance=0.25):  # TODO: magic number
    '''
    Creates a random polygon with approximately n sides of approximately
    length 1. It's guaranteed that the polygon will be convex.
    '''
    pts = [(0, 0), (1, 0)]  # the polygon
    S_a = 0  # the total angle

    # while the newest point is on the left-hand side of pts[0] - pts[1] and
    # while pts[0] is on the left-hand side of the most recent edge and
    # while \Sum a < 2π, continue to add edges
    while True:
        a = rand.gauss(2*pi / n, variance / 2)  # TODO: magic number
        l = rand.gauss(1, variance)
        S_a += a
        last_pt = pts[-1]
        pt = (last_pt[0] + l * cos(S_a), last_pt[1] + l * sin(S_a))
        if S_a <= 2*pi and on_left_hand(pt, (pts[: 2]))\
                and on_left_hand(pts[0], (pts[-1], pt)):
            pts.append(pt)
        else:
            return pts
