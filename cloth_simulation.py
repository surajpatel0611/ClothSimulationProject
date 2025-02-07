# cloth_simulation.py

from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
from direct.task import Task
import sys

# ----------------------------
# Particle and Constraint Data
# ----------------------------
class Particle:
    def __init__(self, pos):
        # Using Panda3D's LPoint3f for positions
        self.pos = LPoint3f(pos)
        self.prev_pos = LPoint3f(pos)  # For Verlet integration

class Constraint:
    def __init__(self, idx1, idx2, rest_length):
        self.idx1 = idx1
        self.idx2 = idx2
        self.rest_length = rest_length

# ----------------------------
# Cloth Simulation Class
# ----------------------------
class ClothSimulation:
    def __init__(self, base):
        self.base = base
        # Cloth grid settings: 21x21 particles forming a 10×10 unit cloth.
        self.num_x = 21
        self.num_y = 21
        self.width = 10.0
        self.height = 10.0
        self.spacing = self.width / (self.num_x - 1)

        self.particles = []   # List of Particle objects.
        self.constraints = [] # List of Constraint objects.

        # We want the cloth to free-fall 1.5 seconds before its bottom touches the sphere.
        # Using s = 0.5 * g * t^2 (with g=9.81 m/s²) we get s ≈ 11.04.
        # With our static sphere’s top at z = 2, we want the cloth’s lowest point to be at 13.04.
        # Since the cloth spans 10 units in height, the center is 5 units above its bottom:
        initial_center_z = 13.04 + 5.0

        # Initialize particles in a grid (lying in the horizontal X-Y plane).
        for j in range(self.num_y):
            for i in range(self.num_x):
                # Center the cloth at (0,0) in X and Y.
                x = (i / (self.num_x - 1)) * self.width - self.width / 2.0
                y = (j / (self.num_y - 1)) * self.height - self.height / 2.0
                z = initial_center_z
                self.particles.append(Particle(LPoint3f(x, y, z)))

        # Create distance constraints between neighboring particles.
        for j in range(self.num_y):
            for i in range(self.num_x):
                idx = j * self.num_x + i
                # Structural constraints (right and down neighbors)
                if i < self.num_x - 1:
                    right_idx = j * self.num_x + (i + 1)
                    self.constraints.append(Constraint(idx, right_idx, self.spacing))
                if j < self.num_y - 1:
                    down_idx = (j + 1) * self.num_x + i
                    self.constraints.append(Constraint(idx, down_idx, self.spacing))
                # Shear constraints (diagonals)
                if i < self.num_x - 1 and j < self.num_y - 1:
                    diag_idx = (j + 1) * self.num_x + (i + 1)
                    self.constraints.append(Constraint(idx, diag_idx, self.spacing * 1.414))
                if i > 0 and j < self.num_y - 1:
                    diag_idx = (j + 1) * self.num_x + (i - 1)
                    self.constraints.append(Constraint(idx, diag_idx, self.spacing * 1.414))

        # Create the renderable mesh (a dynamic geometry that we update each frame)
        self.setup_mesh()

        # Collision sphere (static) parameters.
        self.sphere_center = LPoint3f(0, 0, 0)
        self.sphere_radius = 2.0

        # Simulation parameters.
        self.gravity = LVector3f(0, 0, -9.81)
        self.damping = 0.99              # Damping for the Verlet velocity.
        self.constraint_iterations = 15  # Increased iterations for a stiffer cloth.
        self.friction = 0.8              # Friction coefficient on collision.

    def setup_mesh(self):
        """Creates a dynamic mesh (GeomNode) for the cloth that will be updated each frame."""
        # Use a simple vertex format with 3D positions.
        vformat = GeomVertexFormat.getV3()
        self.vdata = GeomVertexData("cloth", vformat, Geom.UHDynamic)
        self.vdata.setNumRows(len(self.particles))
        self.vwriter = GeomVertexWriter(self.vdata, "vertex")
        for p in self.particles:
            self.vwriter.addData3f(p.pos)

        # Create triangles to render the cloth surface.
        self.prim = GeomTriangles(Geom.UHStatic)
        for j in range(self.num_y - 1):
            for i in range(self.num_x - 1):
                idx = j * self.num_x + i
                idx_right = idx + 1
                idx_down = idx + self.num_x
                idx_down_right = idx_down + 1
                # Each quad is two triangles.
                self.prim.addVertices(idx, idx_down, idx_right)
                self.prim.addVertices(idx_right, idx_down, idx_down_right)
        self.prim.closePrimitive()

        # Build the Geom and attach it to a node.
        self.geom = Geom(self.vdata)
        self.geom.addPrimitive(self.prim)
        self.node = GeomNode("cloth")
        self.node.addGeom(self.geom)
        self.nodePath = self.base.render.attachNewNode(self.node)
        self.nodePath.setTwoSided(True)
        self.nodePath.setColor(1, 1, 1, 1)

    def update(self, dt):
        """Advance the cloth simulation by dt seconds with substepping and iterative collision resolution."""
        substeps = 10  # Use more substeps for finer resolution.
        sub_dt = dt / substeps

        for _ in range(substeps):
            # --- Verlet Integration ---
            for p in self.particles:
                # Compute (damped) velocity from the difference between current and previous positions.
                velocity = (p.pos - p.prev_pos) * self.damping
                temp = LPoint3f(p.pos)
                # Verlet update: new_position = current_position + velocity + acceleration * (sub_dt)^2.
                p.pos = p.pos + velocity + self.gravity * (sub_dt * sub_dt)
                p.prev_pos = temp

            # --- Enforce Constraints ---
            for _ in range(self.constraint_iterations):
                for c in self.constraints:
                    p1 = self.particles[c.idx1]
                    p2 = self.particles[c.idx2]
                    delta = p2.pos - p1.pos
                    dist = delta.length()
                    if dist == 0:
                        continue
                    diff = (dist - c.rest_length) / dist
                    correction = delta * 0.5 * diff
                    p1.pos = p1.pos + correction
                    p2.pos = p2.pos - correction

            # --- Iterative Collision Resolution with the Sphere ---
            # For each particle, perform multiple collision passes until it is no longer penetrating.
            for p in self.particles:
                for _ in range(5):  # Up to 5 iterations per particle per substep.
                    delta = p.pos - self.sphere_center
                    dist = delta.length()
                    if dist < self.sphere_radius:
                        if dist == 0:
                            dist = 0.001  # Avoid division by zero.
                        normal = delta / dist
                        # Project the particle onto the sphere's surface.
                        p.pos = self.sphere_center + normal * self.sphere_radius
                        # Adjust the particle's "velocity" to simulate friction.
                        velocity = p.pos - p.prev_pos
                        normal_component = normal * velocity.dot(normal)
                        tangent_component = velocity - normal_component
                        tangent_component *= self.friction
                        new_velocity = normal_component + tangent_component
                        p.prev_pos = p.pos - new_velocity
                    else:
                        break

        # --- Update the Mesh Vertex Positions ---
        vwriter = GeomVertexWriter(self.vdata, "vertex")
        for p in self.particles:
            vwriter.setData3f(p.pos)
        self.vdata.modifyArray(0)  # Mark the vertex array as changed (optional).

# ----------------------------
# Main Application Class
# ----------------------------
class ClothApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Set the window resolution to 1280x720 (720p) and a black clear color.
        props = WindowProperties()
        props.setSize(1280, 720)
        self.win.requestProperties(props)
        self.win.setClearColor((0, 0, 0, 1))

        # Disable the default mouse-based camera controls so we can implement our own.
        self.disableMouse()
        self.camera.setPos(0, -30, 10)
        self.camera.lookAt(0, 0, 5)

        # Add a simple point light.
        plight = PointLight("plight")
        plight.setColor((1, 1, 1, 1))
        plnp = self.render.attachNewNode(plight)
        plnp.setPos(10, -10, 20)
        self.render.setLight(plnp)

        # Load a sphere model (the collision object).
        try:
            self.sphere = self.loader.loadModel("models/ball")
        except Exception as e:
            print("Could not load 'models/ball'. Loading 'models/smiley' instead.")
            self.sphere = self.loader.loadModel("models/smiley")

        self.sphere.reparentTo(self.render)
        self.sphere.setPos(0, 0, 0)
        self.sphere.setScale(2)  # Radius of 2.

        # Create the cloth simulation.
        self.cloth_sim = ClothSimulation(self)

        # Add the simulation update task.
        self.taskMgr.add(self.simulationTask, "SimulationTask")

        # Set up very simple interactive camera controls:
        # Click and drag with the left mouse button to rotate the camera.
        self.accept("mouse1", self.startRotate)
        self.accept("mouse1-up", self.stopRotate)
        self.isRotating = False
        self.lastMousePos = (0, 0)

    def simulationTask(self, task):
        dt = globalClock.getDt()
        self.cloth_sim.update(dt)
        return Task.cont

    # --- Simple Interactive Camera Controls ---
    def startRotate(self):
        self.isRotating = True
        if self.mouseWatcherNode.hasMouse():
            self.lastMousePos = (self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY())
        self.taskMgr.add(self.rotateCameraTask, "RotateCameraTask")

    def stopRotate(self):
        self.isRotating = False
        self.taskMgr.remove("RotateCameraTask")

    def rotateCameraTask(self, task):
        if self.mouseWatcherNode.hasMouse():
            currentMousePos = (self.mouseWatcherNode.getMouseX(), self.mouseWatcherNode.getMouseY())
            dx = currentMousePos[0] - self.lastMousePos[0]
            dy = currentMousePos[1] - self.lastMousePos[1]
            self.lastMousePos = currentMousePos
            # Adjust the camera's heading and pitch based on mouse movement.
            newH = self.camera.getH() - (dx * 100)
            newP = self.camera.getP() + (dy * 100)
            self.camera.setHpr(newH, newP, 0)
        return Task.cont

# ----------------------------
# Run the Application
# ----------------------------
if __name__ == "__main__":
    app = ClothApp()
    app.run()
