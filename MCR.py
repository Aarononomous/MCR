from math import *
import shapely as sh
from shapely.geometry import *
from shapely.affinity import *
import graphviz as gv
import random as rand
import matplotlib.pyplot as plt
import re


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
        self.overlapped_obstacles = []
        self.graph = []
        self.start = (0.05, 0.05)
        self.goal = (0.95, 0.95)
        # self.field = plt.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

        # All shapes are labeled. A shape can be given an explicit label, or it
        # will be given a numerical label
        self.__label = 0

        if svg:
            # initialize from svg file
            obstacles = MCR.__shapes_from_SVG(svg)
            try:
                for o in obstacles:
                    self.add_obstacle(o)
            except:
                print('Couldn\'t load all shapes')

    def add_obstacle(self, shape):
        '''
        Handles adding new obstacles to the field
        '''

        if not hasattr(shape, 'label'):
            new_label = str(self.__label)
            self.__label += 1
        else:
            new_label = shape.label

        shape.label = new_label
        self.obstacles.append(shape)

        # TODO: clip translated shape to field bounds
        #   - is this necessary? YES
        
        # for every shape that is added, it may intersect with any other
        overlaps = [o_o for o_o in self.overlapped_obstacles if o_o.intersects(shape)]
        
        if overlaps:
            for o_o in overlaps:
                self.overlapped_obstacles.remove(o_o)
                # re-add any part which doesn't overlap
                unoverlapped_part = o_o - shape
                overlapped_part = o_o & shape
                
                if type(unoverlapped_part) == Polygon:
                    unoverlapped_part.label = o_o.label
                    self.overlapped_obstacles.append(unoverlapped_part)
                elif type(unoverlapped_part) == MultiPolygon:
                    for p in unoverlapped_part:
                        p.label = o_o.label
                        self.overlapped_obstacles.append(p)
                # otherwise the new piece is wholly within unoverlapped_part

                if type(overlapped_part) == Polygon:
                    overlapped_part.label = o_o.label + ', ' + new_label
                    self.overlapped_obstacles.append(overlapped_part)
                elif type(overlapped_part) == MultiPolygon:
                    for p in overlapped_part:
                        p.label = o_o.label + ', ' + new_label
                        self.overlapped_obstacles.append(p)
                # otherwise the new piece is wholly within overlapped_part

                # continue adding whatever parts remain
                shape = shape - o_o

            # anything that's not overlapping can be added now
            if type(shape) == Polygon:
                shape.label = new_label
                self.overlapped_obstacles.append(shape)
            elif type(shape) == MultiPolygon:
                for p in shape:
                    p.label = new_label
                    self.overlapped_obstacles.append(p)
        
        else: # nothing overlaps!
            shape.label = new_label
            self.overlapped_obstacles.append(shape)

    def remove_obstacle(self, label):
        '''
        Removes the obstacle labeled 'label'.
        Removes the labeled vertices from the graph and contracts those edges
        '''
        # TODO: everything

        # Displaying the MCR
        pass

    def show_bare_obstacles(self):
        '''
        Outputs the square
        '''
        # add start and goal
        plt.scatter(*zip(self.start, self.goal), color='red')

        # add obstacles
        for obstacle in self.obstacles:
            # TODO: color
            opts = MCR.display_opts.copy()
            if hasattr(obstacle, 'facecolor'):
                opts['facecolor'] = obstacle.facecolor
            if hasattr(obstacle, 'edgecolor'):
                opts['edgecolor'] = obstacle.edgecolor
            poly = plt.Polygon(obstacle.exterior.coords, **opts)
            plt.gca().add_patch(poly)

            if hasattr(obstacle, 'label'):
                r_p = obstacle.representative_point()
                plt.text(r_p.x, r_p.y, obstacle.label,
                         horizontalalignment='center',
                         verticalalignment='center')
            print(obstacle.label + ": " + str(obstacle.area))

        plt.axis()  # set to [0,1]
        plt.show()

    def show_obstacles(self, recreate=True):
        '''
        Outputs the square
        '''
        # add start and goal
        plt.scatter(*zip(self.start, self.goal), color='red')

        # add obstacles
        for obstacle in self.overlapped_obstacles:
            # TODO: color
            opts = MCR.display_opts.copy()
            if hasattr(obstacle, 'facecolor'):
                opts['facecolor'] = obstacle.facecolor
            if hasattr(obstacle, 'edgecolor'):
                opts['edgecolor'] = obstacle.edgecolor
            poly = plt.Polygon(obstacle.exterior.coords, **opts)
            plt.gca().add_patch(poly)

            if hasattr(obstacle, 'label'):
                r_p = obstacle.representative_point()
                plt.text(r_p.x, r_p.y, obstacle.label,
                         horizontalalignment='center',
                         verticalalignment='center')
            
            print(obstacle.label + ": " + str(obstacle.area))
        plt.axis()  # set to [0,1]
        plt.show()

    def create_graph(self):
        '''
        Create the intersection graph. This should be done anytime an object is
        added or removed, and whenever the start/goal locations change.
        Note that this is called by default every time show_obstacles() is run.
        '''
        pass

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

    def __shapes_from_SVG(svg_file):
        try:
            f = open(svg_file)
            svg = f.read()
            f.close()
        except FileNotFoundError:
            print('File {} not found'.format(svg_file))
            return

        # Now start parsing

        shapes = []
        scaled_shapes = []

        #  Viewbox - this will be scaled to 1, 1 eventually
        try:
            viewbox = [float(x)
                       for x in re.findall('viewBox="(.*?)"', svg)[0].split()]
            vb_w = viewbox[2] - viewbox[0]
            vb_h = viewbox[3] - viewbox[1]
        except:
            print('Can\'t find a viewbox!')
            return

        # Rectangles
        rects = re.findall('<rect.*?\/>', svg)
        for r in rects:
            x_ = re.findall('x="([-\d.]+)"', r)
            x = float(x_[0]) if x_ else 0.0

            y_ = re.findall('y="([-\d.]+)"', r)
            y = float(y_[0]) if y_ else 0.0

            h_ = re.findall('height="([\d.]+)"', r)
            h = float(h_[0])

            w_ = re.findall('width="([\d.]+)"', r)
            w = float(w_[0])

            rect = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

            # Transforms
            transform = re.findall('transform="(.*?)"', r)
            if transform:
                t_list = re.findall('\w+\(.+?\)', transform[0])
                for t in reversed(t_list):  # reverse to accumulate (compose)
                    if t.startswith('matrix'):
                        vals = re.findall('\((.+)\)', t)[0]
                        a,d,b,e,xoff,yoff = [float(x) for x in re.split(',?\s+', vals)]
                        rect = affine_transform(
                            rect, [a, b, d, e, xoff, yoff])
                    elif t.startswith('translate'):
                        vals = re.findall('\((.+)\)', t)[0]
                        x, y = (float(x) for x in re.split('\s+,?\s*', vals))

                        rect = translate(rect, x, y)
                    elif t.startswith('scale'):
                        vals = re.findall('\((.+)\)', t)[0]
                        rect = scale(rect,
                                     *[float(x) for x in re.split('\s+,?\s*',
                                                                  vals)],
                                     origin=(0, 0))
                    elif t.startswith('rotate'):
                        val = re.findall('\((.+)\)', t)[0]
                        rect = rotate(rect, float(val), origin=(0, 0))
                    elif t.startswith('skewX'):
                        val = re.findall('\((.+)\)', t)[0]
                        rect = skew(rect, xs=float(vals), origin=(0, 0))
                    elif t.startswith('skewY'):
                        val = re.findall('\((.+?)\)', t)[0]
                        rect = skew(rect, ys=float(vals), origin=(0, 0))

            shapes.append(rect)

        # Polygons
        polygons = re.findall('<polygon.*?\/>', svg)
        for p in polygons:
            polygon = []
            point_list = re.findall('points="(.*?)"', p)
            points = ([float(x) for x in point_list[0].split()])

            points.reverse()  # so that I can pop from the front...
            while points:
                polygon.append((points.pop(), points.pop()))
            polygon.pop()  # remove the doubled last point

            shapes.append(Polygon(polygon))

        # rescale to [1,1]
        # N.b.: the svg viewbox starts at the top left
        for shape in shapes:
            shape = translate(scale(
                shape,
                xfact=1 / vb_w,
                yfact=-1 / vb_h,
                origin=(0, 0)
            ), 0, 1)
            scaled_shapes.append(shape)

        return scaled_shapes

# Solving the thing


def solve_mcr(mcr):
    '''
    Solves the thing.
    '''
    pass


def random_mcr(obstacles=10, scale_factor=0.25):
    '''
    Creates an MCR with random polygonal obstacles
    '''
    mcr = MCR()
    pts = poisson_2d(obstacles)
    new_obs = []

    for i in range(obstacles):
        r = scale(random_shape(), scale_factor, scale_factor)
        x, y = pts.pop()
        minx, miny, _, _ = r.bounds
        new_obs.append(translate(r, x - minx, y - miny))

    # rescale again, to fit all shapes in the
    bounds = [o.bounds for o in new_obs]
    max_x = max([b[2] for b in bounds])
    max_y = max([b[3] for b in bounds])

    for obstacle in new_obs:
        mcr.add_obstacle(scale(obstacle, 1 / max_x, 1 / max_y, origin=(0, 0)))

    return mcr


def random_shape(max_sides=6):
    '''
    Creates a random convex polygon with vertices between [0,1] x [0,1].
    '''
    shape = approx_ngon(rand.randrange(3, max_sides + 1))

    # rotate it randomly
    return rotate(shape, rand.uniform(0, 2*pi),
                  use_radians=True,
                  origin='centroid')


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
            return Polygon(pts)


def poisson_2d(k):
    '''
    Returns an array of k (x, y) tuples evenly spaced across [0, 1] x [0, 1]
    Note that there are tweaks to the distribution that make it unreliable
    as an _actual_ Poisson distribution.
    '''
    xs = [0]
    ys = [0]

    for _ in range(k-1):
        xs.append((xs[-1] + rand.expovariate(k+1)))
        ys.append((ys[-1] + rand.expovariate(k+1)))

    max_x = max(xs)
    max_y = max(ys)

    xs = [x / max_x for x in xs]
    ys = [y / max_y for y in ys]

    rand.shuffle(ys)
    return list(zip(xs, ys))
