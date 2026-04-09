Offset Geometric Contact
ANKA HE CHEN, University of Utah, NVIDIA, USA
JERRY HSU, University of Utah, USA
ZIHENG LIU, University of Utah, USA
MILES MACKLIN, NVIDIA, New Zealand
YIN YANG, University of Utah, USA
CEM YUKSEL, University of Utah, USA

Fig. 1. Example simulation results using our penetration-free contact handling method. Our method is robust in the presence of challenging
contact scenarios, and can be easily integrated with existing solvers such as Vertex Block Descent [Chen et al. 2024b], as shown here.

We present a novel contact model, termed Offset Geometric Contact (OGC),
for guaranteed penetration-free simulation of codimensional objects with
minimal computational overhead. Our method is based on constructing
a volumetric shape by offsetting each face along its normal direction, en-
suring orthogonal contact forces, thus allows large contact radius without
artifacts. We compute vertex-specific displacement bounds to guarantee
penetration-free simulation, which improves convergence and avoids the
need for expensive continuous collision detection. Our method relies solely
on massively parallel local operations, avoiding global synchronization and
enabling efficient GPU implementation. Experiments demonstrate real-time,

Authors’ addresses: Anka He Chen, ankachan92@gmail.com, University of Utah,
NVIDIA, Kirkland, WA, USA; Jerry Hsu, jerry060599@gmail.com, University of Utah,
Salt Lake City, UT, USA; Ziheng Liu, ziheng.liu@utah.edu, University of Utah, Salt
Lake City, UT, USA; Miles Macklin, mmacklin@nvidia.com, NVIDIA, Auckland, New
Zealand; Yin Yang, yangzzzy@gmail.com, University of Utah, Salt Lake City, UT, USA;
Cem Yuksel, cem@cemyuksel.com, University of Utah, Salt Lake City, UT, USA.

This work is licensed under a Creative Commons Attribution 4.0 International License.
© 2025 Copyright held by the owner/author(s).
0730-0301/2025/8-ART
https://doi.org/10.1145/3731205

large-scale simulations with performance more than two orders of magni-
tude faster than prior methods while maintaining consistent computational
budgets.

CCS Concepts: • Computing methodologies → Physical simulation;
Collision detection.

Additional Key Words and Phrases: physics-based simulation, elastic body,
rigid body, time integration

ACM Reference Format:
Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuk-
sel. 2025. Offset Geometric Contact. ACM Trans. Graph. 44, 4 (August 2025),
21 pages. https://doi.org/10.1145/3731205

1

INTRODUCTION

Penetration-free simulation is essential for many graphics applica-
tions, especially those involving codimensional models, as pene-
tration can cause significant artifacts or even break the simulation.
Moreover, once penetration occurs, it is particularly challenging
to resolve, especially in codimensional models. Despite significant
advancements in penetration-free simulations, such as Incremental
Potential Contact (IPC) [Li et al. 2020], a critical problem remains:

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

2

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

(a) IPC

(b) OGC

(c) IPC with large
contact radius

(d) IPC with small
contact radius

(e) OGC with large
contact radius

Fig. 2. Illustrating both the artifact produced by the IPC contact model and its underlying cause. We simulate a twisted square cloth with 40K
vertices and 79.2K faces, each side measuring 1 meter, rotated by half a circle. The simulation is conducted using both the IPC and OGC models,
with a fixed contact radius of 5 mm. (a) and (b) show the final states of the cloth using the IPC and our OGC contact models, respectively. Panels
(c) and (d) depict the IPC contact model, which is equivalent to offsetting the face in all directions to form a capsule-like shape. The dashed line
marks the boundary of this shape, black dots represent contact points, and the colored arrows indicate the forces exerted from or onto the face with
corresponding color. Panel (e) visualizes our proposed contact model, where the dashed lines mark the boundaries of blocks from corresponding
faces with the same color.

the computational cost. IPC-based simulators and their derivatives
are usually orders of magnitude more expensive than alternative
methods that do not provide such a guarantee. Moreover, the compu-
tational cost of those methods depends on each time step’s state and
is highly uneven. These issues prevent penetration-free simulations
from being used in many applications, especially those requiring
real-time performance.

The problem comes from two main factors: the collision energy is
very stiff, making convergence difficult, and expensive procedures
such as line search and collision detection must be incorporated
into every iteration of the simulation to ensure penetration-free
conditions.

The stiffness arises from the necessity of preventing penetration.
To ensure the contact force is always strong enough to push objects
apart, it must be able to become infinitely large as objects approach
each other. This requires the contact force to transition from zero to
infinite within the contact radius. This issue becomes particularly
severe when the contact radius is very small.

In this work, we identify a geometric limitation of IPC: the re-
sultant normal contact force may not always be orthogonal to the
surface (see Figure 2c), potentially leading to artifacts (see Figure 2a).
To address this, IPC employs a scheme that adaptively reduces the
contact radius during the optimization process until it becomes
extremely small. However, this further increases the stiffness of the
contact energy.

IPC uses continuous collision detection (CCD) based technique to
prevent collision. This technique applies a CCD on the optimization
step provided by the simulation solver, and culls it before the earliest
intersection happens. For GPU implementation, this procedure is a
global operation that requires synchronization and hinders paral-
lelism. CCD is applied at every iteration, which is very expensive.
Moreover, the CCD-based intersection filter halts the global opti-
mization step where the earliest intersection happens, meaning a
local intersection stops the progress of all other points, even if those
points are still far from intersecting. This could reduce the solver’s
efficiency, since each iteration can be computationally expensive,

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

and the shortened optimization step induced by CCD results in more
iterations.

We propose a novel method, termed Offset Geometric Contact
(OGC), to ensure guaranteed penetration-free simulation for codi-
mensional objects, achieving this with only a minimal and near-
constant overhead added to simulators that do not offer such a
guarantee. Our method includes the following:

(1) A novel contact formulation based on offsetting the surface as
a whole instead of offsetting each primitive separately. This
guarantees that the contact force will always be orthogonal
to the face it applies on and never cause a stretching artifact.
This allows the usage of a relatively larger contact distance,
making the contact significantly less stiff.

(2) A different approach to guarantee penetration-free simula-
tion, which does not require CCD. This is enabled by our
contact model, which accommodates a larger contact radius.
Specifically, it computes an individual maximum displace-
ment bound for each vertex concurrently with collision de-
tection, which adds a minimal overhead. For a vertex far from
contact, its bound will be larger, allowing it to fully utilize the
optimization step given by the solver, significantly improving
the convergence.

Moreover, both the computation of our contact force and the
penetration-prevention technique are local operations, which are
massively parallel and do not require global synchronization. When
combined with a fully parallel solver such as Vertex Block Descent
[Chen et al. 2024b], our method can be very efficient on GPUs,
providing real-time, large-scale, penetration-free simulations such
as results shown in Figure 1. Our tests show that our method can be
more than two orders of magnitude faster than IPC-based simulation,
and can use a near-constant computational budget by using a fixed
iteration count.

2 BACKGROUND

Contact occurs between two surfaces. For each point on one sur-
face that contacts another surface, it is subjected to a contact force
from the opposing surface. This force generally consists of two com-
ponents: a normal force, which acts perpendicular to the contact
surface, and a friction force, which acts parallel to the contact sur-
face and is linearly related to the normal force. The normal force is
a conservative force, while the friction force is not. Therefore we
can write normal force as the negative gradient of a normal contact
energy 𝐸𝑛. The formulation of 𝐸𝑛 differentiates different contact
models.

2.1 Basic Contact Model

In physics-based simulation, surfaces are represented by polygonal
mesh, denoted by 𝑀 = {V, E, T }, where V, E, T denote the set
of vertices (0-face), edges (1-face) and facets (2-face, e.g., triangles,
quadrilaterals, etc.), respectively. It is important to note that 𝑀 can
consist of multiple connected components, which accommodates the
presence of multiple models. Therefore, without loss of generality,
in the following discussion, we assume the presence of a single mesh
𝑀. We denote 𝑋 ∈ R𝐾 ×3 as the stacked positions of all the vertices,
where 𝐾 = |V |. The position of vertex 𝑣 is represented as x𝑣.

We define the normal contact energy as a function of the dis-
tance between two primitives, namely, between vertex and facets
or between two edges. Based on the first law of friction, the contact
force can be computed in such order: computing the normal force
first, and then calculating the friction force using the friction coef-
ficient. Therefore, the normal contact force plays a key role in the
computation of contact force.

We start with the vertex-facet contact pair. Given a mesh 𝑀, the
𝑛 of 𝑀 is usually defined in the following

normal contact energy 𝐸𝑣𝑓
form:

𝐸𝑣𝑓
𝑛 (𝑀, 𝑟 ) =

∑︁

𝑎∈ F (𝑣)

𝑔(𝑑𝑖𝑠 (x𝑣, 𝑎), 𝑟 ) ,

(1)

where F (𝑣) is the set of all the faces that are in contact with 𝑣,
𝑑𝑖𝑠 (x𝑣, 𝑎) is the function computing the distance between vertex
x𝑣 position and a face 𝑎, 𝑟 is contact radius, and 𝑔 is a nonlinear
function. We define the closest point from x𝑣 to 𝑎 as:

c(x𝑣, 𝑎) = arg min

||x − x𝑣 ||.

(2)

x∈𝑎
Therefore 𝑑𝑖𝑠 (x, 𝑎) = ||x − c(x, 𝑎)||. Since 𝑔 is just a scalar function,
it does not change the direction of the force. Therefore, the contact
force between 𝑓 and x, is always parallel to x − c(x, 𝑎).

The different choices of 𝑔 and F (𝑣) result in different collision
energies. We start with discussing the choice of F (𝑣), termed contact
face set.

2.2 Contact Face Set

One common choice of F is:

FIPC (𝑣) = {𝑡 ∈ T |𝑑𝑖𝑠 (x𝑣, 𝑡) < 𝑟, 𝑣 ⊄ 𝑡 } .
(3)
Namely, FIPC (𝑣) takes all the facets who do not include 𝑣 and whose
distance to 𝑣 is less than the contact radius 𝑟 . This contact face set
is employed by the well-known Incremental Potential Contact [Li
et al. 2020]. Intuitively, this formulation is like inflating the facets in

Offset Geometric Contact

•

3

(a) FIPC

(b) FIPC

(c) FSDF

Fig. 3. 2D illustration of different contact face sets and the normal
contact force derived from them. x is the position of the vertex of
the vertex-facet (v-f) contact pair, and the black circle visualizes the
contact radius of point x. The colored line segments represent facets,
and the colored arrows represent the normal contact force applied to
the facets of the same color. (a, b) visualize FIPC. (c) visualizes FSDF,
where the dashed black line represents the bisector of those two facets.

all directions, forming volumetric shapes as illustrated in Figure 2c.
A facet contact with 𝑣 if 𝑣 is located inside the inflated facets.

There are two major problems with this energy formulation. The
first one is illustrated in Figure 2c and Figure 3a: the "normal" contact
force applied on the green and blue facets is not perpendicular
to them, causing a stretching force component on the tangential
plane. Another problem is that it pushes x’s topological neighbors
away, as illustrated in Figure 3b. While it is possible to ignore the
contact between 𝑣 and its neighboring facets, the problem still exists
between a vertex and its 2-ring neighbors. Unfortunately, we cannot
filter out these contacts as this can cause penetration. As illustrated
in Figure 2a, when a large contact radius is used, those problems
can cause serious artifact including stretching and oscillating.

IPC mitigates those problems by dynamically adjusting the con-
tact radius to a very small value (Figure 2d), to the extent where
it is nearly impossible for a vertex to be in contact with multiple
adjacent facets. However, choosing a small contact radius leads to
other problems including numerical issues such as the stiffness of
the contact energy. Additionally, IPC’s CCD-aware line search to
avoid penetration limits the optimization step size when 𝑣 is close
to the contacting surface, as smaller distances trigger earlier pene-
tration. Moreover, since the contact radius is so small, the surface
region that stops 𝑣 may not be in contact with 𝑣 when computing
the optimization direction. Therefore, the optimization direction
may not be a direction that separates those colliding pairs, which
further restricts its movement per iteration. These factors contribute
to IPC’s high iteration count for convergence.

An alternative selection for F (x) is to select only the closest

facet:

FSDF (𝑣) = {𝑡 |𝑡 = arg min

𝑡 ∈ T

𝑑𝑖𝑠 (x𝑣, 𝑡) and 𝑑𝑖𝑠 (x𝑣, 𝑡) < 𝑟 }

(4)

This contact face set is the basis of many signed distance field (SDF)
based collision energy. The main advantage of this approach is
that it guarantees the normal force is always perpendicular to the
contacting point on 𝑀, a natural property of the closest point on a
smooth manifold as established by the Hilbert Projection Theorem.
However, it has some serious problems due to its limiting the
number of a vertex’s contacting facets to only one. This restriction

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱𝐱4

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

could impede the convergence of the problem because it fails to
generate a sufficient number of collision pairs to push away vertex-
facet pairs that are close enough. Instead, it results in the vertex
constantly switching with a few facets. This is illustrated in Figure 3c,
where the point x alternates between contacting the red facet and
the green facet, oscillating along their bisector. To make matters
worse, this formulation can not resolve self-intersection. Since 𝑣
is part of 𝑀, the SDF at x𝑣 is 0, and this information will not help
resolve 𝑣’s contact with 𝑀. That is why this contact face set is
usually used to handle contact of static objects.

2.3 Related Works

Penetration-free simulation is a recent breakthrough in the physics-
based simulation community. The simulation of deformable bodies
is usually done by minimizing the implicit integration equation,
with the collisions modeled as potential energies, or additionally,
as constraints to the minimization problem. To strictly guarantee
penetration-free, the collision must be formulated as non-compliable
constraints. In the physics-based simulation community, those non-
compliable constraints are usually enforced through two groups
of methods: the line search based methods and the trust region
methods.

2.3.1 Line Search Based Methods. The incremental potential con-
tact (IPC) method [Li et al. 2020] models collision energy using a
logarithmic function that approaches infinity as primitives come
closer, ensuring that contact forces overcome other forces to pre-
vent penetration. To enforce the penetration-free condition, IPC
requires optimization to halt before the earliest time of impact (TOI),
determined via CCD-aware line search. The process involves iter-
atively recomputing contact relationships, descent directions, and
CCD checks until convergence. Li et al. [2021] later extended this
IPC collision model to codimensional objects, e.g., elastic rods and
surfaces. This includes several novel improvements: modeling thick-
ness, a generalized CCD that adapts to this thickness modeling, and
another barrier function to limit the stretching.

CCD-aware line search requires linear motion at each optimiza-
tion iteration, which is not satisfied for systems with rotational
components like rigid body dynamics. Ferguson et al. [2021] ad-
dressed this by dividing rotational motion into small linear segments
for CCD, but this incurs more computation steps. Lan et al. [2022a]
improved this by using affine transformations instead of SE(3) move-
ments, turning rotational motion into linear affine motion. This
approach eliminates the need for multiple segments, requiring only
one CCD application per step, greatly enhancing simulation effi-
ciency.

Various methods have been employed to enhance IPC simula-
tion efficiency [Guo et al. 2024; Lan et al. 2024; Shen et al. 2024;
Wu et al. 2022]. Lan et al. [2022b] replaced IPC’s Newtonian solver
with projective dynamics, enabling penetration-free GPU simula-
tions by reformulating IPC’s barrier constraint with projected target
positions. Lan et al. [2023] introduced a block coordinate descent
technique with element-based Gauss-Seidel iteration and local CCD
to reduce computational costs. Huang et al. [2024b] developed a
GPU-accelerated Gauss-Newton method to accelerate simulations

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

using barrier contact energy. Lan et al. [2021] simplified original ge-
ometry with standard shapes to reduce collision pairs and speed up
simulations, sacrificing fine geometric details. Ando [2024] replaces
the logarithmic barrier with a cubic one to reduce the stiffness of
the contact energy.

2.3.2 Trust Region Based Methods. In numerical optimization, a
trust region defines a subset of the domain where the objective
function is approximated, typically using a quadratic model [No-
cedal and Wright 1999]. The region adapts dynamically: expand-
ing if the model proves accurate and contracting if it does not,
enabling efficient optimization. Trust-region methods can be consid-
ered somewhat complementary to line-search methods: trust-region
approaches initially determine a step size (the dimensions of the
trust region) and subsequently select a step direction. In contrast,
line-search methods start by choosing a step direction and then
decide on the step size.

It is known that trust-region based methods are better suited for
constrained optimization problems where constraint satisfaction
is critical [Pavlakos et al. 2019; Yuan 2015]. Constraints can be
incorporated directly into the trust region formulation [Burke 1992;
Moré 1983], which are usually linear and convex [Conn et al. 1988],
[Burke et al. 1990].

However, trust region methods for enforcing penetration-free
constraints have not been extensively explored in the simulation
community. Unlike IPC, which combines CCD with line search to
enforce penetration constraints, trust-region methods use discrete
collision detection (DCD) to define per-vertex (or per-rigid-body)
trust regions, constraining movements to prevent penetration.

This kind of idea was initially explored in the context of rigid
body dynamics, termed Conservative Advancement. Zhang et al.
[2006] uses the extremal vertex query to find a directed motion
bound for an object moving with constant translational and rota-
tional velocities. Tang et al. [2009] extends this kind of method to
triangulated models and makes no assumption about the underlying
geometry and topology. Chen et al. [2024a] applied a trust-region
based scheme to filter eigenvalues in the system Hessian for New-
ton’s method.

This idea has also been investigated in the context of cloth simu-
lation. Wu et al. [2020] identified a necessary vertex displacement
constraint to prevent cloth from self-intersecting, thus ensuring the
avoidance of self-penetration at all times. Wang et al. [2023] utilized
this constraint within a step-and-project process to facilitate fast
and realistic simulation.

However, the development of a trust region-based simulation sys-
tem that incorporates barrier collision energy and ensures numerical
convergence remains an open area of research.

2.3.3 Offset Geometry. Offset geometries, also known as polygon
offsets or Minkowski dilation, are geometric constructions where
a polygon is expanded or contracted by a specified distance. The
geometry 𝑀’s offset geometry with distance 𝑟 is defined as:

𝑀+𝑟 = {x ∈ R𝑁 |𝑑𝑖𝑠 (x, 𝑀) < 𝑟 }

(5)

The boundary of this offset geometry can be computed through
various methods, including winding number based methods [Chen
and McMains 2005], Voronoi Diagram-Based Methods [Bo 2010],

Fig. 4. (a) original geometry 𝑀; (b) conventional offset geometry 𝑀+𝑟 ;
(c) our intersection-aware manifold U.

straight skeleton based methods [Aichholzer et al. 1996; Huber 2018],
polygonal annulus based methods [Barequet and Goryachev 2014].
However, those methods mainly focus on 2D polygons instead
of 3D polyhedral meshes. Moreover, The conventional concept of
offset geometry presents significant challenges when applied to
contact modeling. Specifically, traditional offset methods often fail
to accurately represent self-intersections and overlapping regions
within the geometry, as illustrated in Figure 4b. The offset geom-
etry given by Equation 5 will "merge" parts that are separated in
the original manifold. This occurs when a point’s distance to two
separate parts of 𝑀 is both less than 𝑟 . This limitation can lead to
inaccuracies in simulating contact interactions, as the model may
not correctly account for multiple layers of contact and self contacts.
Ideally, the offset geometry should be aware that there are 2
overlapping layers, as illustrated in Figure 4c, and the point in
the overlapping area should be subject to contact forces from the
opposite side. The dimensionality of the offset geometry in Figure 4c
has been lifted to 𝑁 (the dimensionality of the immersion space),
and therefore is not codimensional anymore. Intuitively, we can
determine the layers of intersection, and compute the penetration
depth using a method akin to [Chen et al. 2023] to compute contact
energy.

2.3.4 Gauss Map. Banchoff [1967, 1970] extended the Gauss-
Bonnet theorem to polyhedral surfaces by introducing a method to
compute curvature at vertices using the Gauss map. Brehm and
Kühnel [1982] further expressed curvature measures in terms of
the number of critical points. Horn [1984] introduced the concept
of Extended Gaussian Images (EGI) for object recognition by
projecting the normal vectors of a polyhedron’s faces onto a sphere,
assigning densities proportional to the corresponding face areas.
Little [1985] proposed a variation of EGI where normal vector
lengths are proportional to face areas, investigating its uniqueness
for convex polyhedra and its application in reconstructing objects
using the Minkowski theorem. This approach requires defining face
orientations, known as combinatorial types, and an iterative process
for 3D reconstruction. Cohen et al. [1998] estimated curvature
for polygonal surfaces using normal cycles at vertices, edges, and
triangles, providing error bounds for discrete surfaces derived from
restricted Delaunay triangulation. Building on these, Echeverria
[2007] propose a novel approach to curvature measurement that
distinguishes positive and negative components, enabling accurate
vertex characterization. Their Polyhedral Gauss Map directly
correlates normal vectors from the polygonal mesh, reflecting
vertex geometry and their local neighborhoods more precisely.

Offset Geometric Contact

•

5

3 OFFSET GEOMETRIC CONTACT

We propose a new normal contact force model that has the following
properties:

• Orthogonality: our normal contact force is always orthog-
onal to the contact surface. It will not create a stretching
artifact even with a large contact radius.

• Large Contact Radius: The contact radius can be arbitrarily

large and still not cause artifacts.

• Multiple Contacts within Contact Radius: multiple prim-
itives within the contact radius can affect x, which allows a
repulsive force to be generated for an arbitrary number of
close-by primitive pairs.

• Self-Intersection Aware: this contact force can identify
arbitrary layers of self-intersection, and effectively resolve
them.

Our normal contact force is derived from an offset geometry of the
original mesh, hence the name Offset Geometric Contact (OGC). We
construct the building blocks of this offset geometry by offsetting a
face along all its normal directions, which is given by its Polyhedral
Gauss Map (Section 3.1). We further provide constructive definitions
of those building blocks to determine whether a point is inside
(Section 3.2, 3.5), and define our own contact face set (Section 3.3).
Subsequently, we derive the penetration in the offset geometry
(Section 3.4) and introduce a new activation function to formulate
our normal contact energy (Section 3.6). At last we propose our
approach to guarantee penetration-free simulation (Section 3.7),
and compare it to IPC’s method (Section 3.8).

3.1 Polyhedral Gauss Map

We build the offset geometry using a way akin to tetmesh: offset
each face individually and use them as the building blocks of the
offset geometry. From the previous discussion, it is evident that the
normal contact force is always parallel to x𝑣 −c(x𝑣, 𝑎), see Equation 1.
Since our goal is to achieve an orthogonal normal contact force,
intuitively, for any face 𝑎 we can design its building block such that
it contacts only points that generate contact forces parallel to the
normal at the contact point. In other words, it should only contact
points x ∈ R𝑁 that satisfies x − c(x, 𝑎) being parallel to the normal
of c(x, 𝑎).

On a smooth surface, enforcing such a condition is relatively
straightforward because each point on the surface has a unique nor-
mal. However, on a polyhedral surface, normals are not so trivially
defined, introducing additional complexity. These considerations
naturally lead to the concept of the Polyhedral Gauss Map (PGM).
Polyhedral Gauss Map is an analogy of a Gauss Map on a poly-
hedral surface, mapping a point on a polyhedral surface to their
associated normals. The key difference is that a point on a poly-
hedral surface can have multiple normals, as opposed to those on
smooth surfaces that only have one. The points that have more than
one normal lies on faces with dimensionality less than 2, such as
the vertices and edges of a 3D polyhedral mesh.

Echeverria [2007] proposed a form of Polyhedral Gauss Map
for vertices on a polyhedral mesh. Since all the normals are unit
vectors, we can draw them on a unit sphere. As illustrated in Figure 5,
Echeverria [2007] classifies vertices into three types: convex, mixed,

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

(a)𝑀𝑀(b) 𝑀𝑀+𝑟𝑟(c)6

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

(a) Convex Vertex

(b) Mixed Vertex

(c) Saddle Vertex

Fig. 5. Gauss map of different types of vertices (top row) and their
spherical indicatrix (bottom row). The area with the pink color repre-
sents the local geometry of the triangular mesh, and the solid green
area represents the normals where the point maps to.

and saddle, based on the geometry of their neighborhood which
are visualized using the pink surface. Following the terminology of
[Echeverria 2007], the neighbor area is called 𝑠𝑡𝑎𝑟 (𝑣).

As the name indicates, a convex vertex 𝑣 is one whose neigh-
borhood is convex. The Gauss Map of a convex vertex is relatively
straightforward, as shown in Figure 5a as the green volume. Intu-
itively, this volume corresponds to the set of points that are closer
to 𝑣 than any other points in 𝑠𝑡𝑎𝑟 (𝑣).

A mixed vertex 𝑣 is one that remains a vertex of the convex
hull of 𝑠𝑡𝑎𝑟 (𝑣). As shown in Figure 5b, the Gauss Map of a mixed
vertex consists of two distinct types of regions. one corresponds to
the positive curvature, as visualized by the green volume, which
corresponds to its Gauss map as a vertex of the convex hull of 𝑠𝑡𝑎𝑟 (𝑣).
The red volume corresponds to regions of negative curvature, with
each negative segment associated with a non-convex neighboring
edge.

The final type is the saddle vertex, which lies within the interior
of the convex hull of 𝑠𝑡𝑎𝑟 (𝑣). The Gauss Map of a saddle vertex is
an empty set because such vertices exhibit no angular deficit.

Echeverria [2007] provided an intuitive explanation of their pro-
posed Gauss Map: plot all the normals of the neighboring facets
of a vertex onto the unit sphere, resulting in a set of points. Con-
nect these points in the counter-clockwise order of the neighboring
facets, following great circles, to form a polygon on the unit sphere.
In this representation:

• A neighboring facet corresponds to a vertex of the polygon.
• A neighboring edge corresponds to an arc-edge of the poly-

gon, which is perpendicular to the neighboring edge.
• The vertex itself corresponds to the polygon as a whole.

For a mixed vertex, due to its concavity, the polygon may contain
inverted areas. These inverted areas represent regions of negative
curvature.

The motivation for Echeverria [2007] to define the Polyhedral
Gauss Map is to extend Gauss–Bonnet theorem to a polyhedral
surface. That is why they only proposed the Gauss map of vertices

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

(a) Face Normals

(b) Edge Normals

(c) Face Normal Spherical Indicatrix

(d) Edge Normals Spherical Indicatrix

Fig. 6. Illustration of Gauss Maps of a facet (triangle) and an edge. In
the top row, the area with the pink color represents the local geometry
of the triangular mesh, and the solid green area represents the normals
to which the point maps. In the bottom row, we show the spherical
indicatrix, i.e. visualize the corresponding point’s normal on a unit
sphere.

because only vertices are integrated. However, in our case, we also
need to define the Polyhedral Gauss Map of edges and facets. Since
all points in the interior of a face share the same normal, we can
instead define the Gauss Map at the level of faces. We denote the
Gauss Map of a face 𝑎 as N𝑎.

The Gauss Map for points on facets and edges is relatively straight-
forward. As shown in Figure 6c, all points on the interior of a facet
map to a single point on the unit sphere, corresponding to the nor-
mal of that face, see Figure 6a. Conversely, a point on the interior of
an edge corresponds to multiple normals, maps to an arc on the unit
sphere see Figure 6b and Figure 6d. This is because that the mesh is
flats on all the triangles, and it "turns" on edges, and the normals of
each edge correspond to how much angle the surface turns.

For vertex Gauss maps, we adopt a slightly modified definition
tailored to our use case in contact detection. Unlike Echeverria
[2007], which defines the Gauss Map from a curvature point of
view, we follow a discrete interpretation of the Hilbert Projection
Theorem. Specifically, if n is the normal at point y on 𝑀, then for
any point x satisfies x = 𝑤n + y, 𝑤 > 0, y must be closer to x than
its local neighborhood. This perspective alters the Gauss Map of
mixed vertices to only include the region corresponding to positive
curvature, i.e., the green area in Figure 5b. This adjustment not
only simplifies the computation but also ensures that when a vertex
contacts a point, it is the locally closest point to that point.

We also make some specific treatments to a facet’s 𝑡 (e.g., triangles)

Gauss Map, we define:

N𝑡 = {n𝑡 , −n𝑡 },

(6)

where n𝑡 is the normal of 𝑡. In other words, we offset a facet to both
of its sides.

3.2 Constructive Definition of Blocks

Now we have clarified the definition of normals on a discrete surface.
We can offset points on 𝑀 along their normal directions to construct
building blocks of the offset geometry.

For a face 𝑎 ∈ 𝑀, we offset its interior points to construct the

fundamental building blocks of the offset geometry:
𝑈𝑎 = {x ∈ R𝑁 |x = y + 𝑤𝑟 n𝑎, where y ∈ 𝑎◦, n𝑎 ∈ N𝑎, 𝑤 ∈ [0, 1]}
(7)
where 𝑟 > 0 is the offset radius, and 𝑎◦ denotes the interior of face
𝑎. For convenience we let 𝑣 ◦ = 𝑣. We only offset the interior of a
face because the boundary points are actually points on a lower
dimensional face.

We refer to 𝑈𝑎 as the block derived from face 𝑎, it serves as a
fundamental building block of the offset manifold. The definition
provided in Equation 7 reflects the essence of these blocks but is
computationally challenging to implement. Fortunately, the earlier
specialized treatment of mixed vertices enables this constructive
formulation of blocks.

A vertex block of a vertex 𝑣 ∈ V is the region enclosed by all
planes passing through 𝑣 and perpendicular to its convex neighbor-
ing edges (i.e., the edges that remain as edges in the convex hull
of 𝑠𝑡𝑎𝑟 (𝑣)). As illustrated in Figure 7d, its shape resembles a ball
with radius 𝑟 , cut by multiple planes that are perpendicular to its
neighboring edges. The constructive definition of 𝑈𝑣 is as follows:
𝑈𝑣 = {x ∈ R𝑁 | ||x − x𝑣 || ≤ 𝑟, (x − x𝑣)(x𝑣 − x′
𝑣) ≥ 0 for 𝑣 ′ ∈ V𝑣 },
(8)
where x𝑣 denote the position of vertex 𝑣, and V𝑣 is the set of all
vertices adjacent to 𝑣. Note that we do not differentiate non-convex
neighboring edges. This is because the planes associated with non-
convex neighboring edges only intersect with 𝑈𝑣 at 𝑣 and, therefore,
do not influence the shape of 𝑈𝑣 in the definition given by Equa-
tion 8.

An edge block of an edge 𝑒 ∈ E is illustrated in Figure 7b. It is a
cylinder with a radius of 𝑟 being cut by 4 planes: 2 being perpendic-
ular to the edge and 2 being perpendicular to each of the edge’s two
neighboring faces. The constructive definition of 𝑈𝑒 is as follows:

Offset Geometric Contact

•

7

(a) Vertex Block

(b) Edge Block

(c) Boundary Edge Block

(d) Facet Block

Fig. 7. Illustration of blocks corresponding to different types of faces.
The regions shaded in light green represent the blocks, while the areas in
solid green indicate the faces associated with these blocks. Boundaries
marked with dashed lines are open, whereas those with solid colors
are closed.

a half-cylinder-like block for the boundary edge, as illustrated in
Figure 7c.

The block of a face 𝑡 ∈ T is straightforward: it is formed by off-
setting the face along its normal direction by a distance 𝑟 , forming a
prism (see Figure 7a). The constructive definition of 𝑈𝑡 is as follows:

𝑈𝑡 ={x ∈ R𝑁 |

𝑑𝑖𝑠 (x, 𝑡) ≤ 𝑟,
(p(x1, x2, x3) − x)(p(x1, x2, x3) − x3) > 0,
(p(x2, x3, x1) − x)(p(x2, x3, x1) − x1) > 0,
(p(x1, x3, x2) − x)(p(x1, x3, x2) − x2) > 0
where x1, x2, x3 are the three vertices of 𝑡 }
Another advantage of this constructive definition is that it nat-
urally gives the definition of boundary edges and vertices, whose
normals are not defined by Echeverria [2007].

(10)

𝑈𝑒 ={x ∈ R𝑁 |

3.3 Contact Face Set

𝑑𝑖𝑠 (x, 𝑒) ≤ 𝑟,
(x − x𝑣𝑒,1 )(x𝑣𝑒,2 − x𝑣𝑒,1 ) > 0,
(x − x𝑣𝑒,2 )(x𝑣𝑒,1 − x𝑣𝑒,2 ) > 0,
(x − p(x(𝑣𝑒,1), x(𝑣𝑒,2), x𝑣𝑒,next )·
, x𝑣𝑒,2
(p(x𝑣𝑒,1
for 𝑣𝑒,next ∈ V𝑒 }

, x𝑣𝑒,next ) − x𝑣𝑒,next ) ≥ 0

(9)

where 𝑣𝑒,1 and 𝑣𝑒,2 are the two vertices of 𝑒, p(x1, x2, x3) computes
the perpendicular foot for x3 on the line defined by x1, x2, and V𝑒
is the set of the vertices that share a facet with 𝑒. It is worth noting
that since 𝑀 is a manifold, V𝑒 can contain at most two vertices,
each belonging to one of 𝑒’s neighboring faces. Additionally, when
𝑒 is a boundary edge, V𝑒 contains only one vertex, resulting in

U = {𝑈𝑎 | 𝑎 ∈ V ∪ E ∪ T }
(11)
We call U the Intersection Aware Offset Geometry of 𝑀. The Inter-
section Aware Offset Geometry serves as an analogy to volumetric
meshes. The elements in U act as a building block of the geome-
try, as tetrahedron to tetmesh. A point can intersect with multiple
𝑈𝑎 ∈ U, this is how we know it has multi-layers of intersection
with U.

Based on U, we can define a new type of Contact Face Set as:
FOGC (𝑣) = {𝑎 | x𝑣 ∈ 𝑈𝑎, 𝑣 ⊄ 𝑎}.
(12)
We refer to |FOGC (x𝑣)| as the number of layers of intersections for
𝑣.

For a point x ∈ R𝑁 and a face 𝑎, if x ∈ 𝑈𝑎, there must exist
y ∈ 𝑎◦ such that x = y + 𝑤𝑟 n, where n ∈ N𝑎, 𝑤 ∈ [0, 1]. Since 𝑎 is

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

8

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

a linear element, y = c(x, 𝑎) must hold, which means x − c(x, 𝑎) =
𝑤𝑟 n. Therefore, our selection of Contact Face Set FOGC guarantees
that each point will only contact faces that generate orthogonal
normal contact force. In fact, our contact model has the following
advantages:

• Orthogonality: for each point x ∈ 𝑈𝑎, (x − c(x, 𝑎)) ⊥ 𝑎 in a

discrete sense.

• Local Exclusiveness: if 𝑎 ⊂ 𝑏, 𝑈𝑎 ∩ 𝑈𝑏 = ∅.
• Covering 𝑀+𝑟 : (cid:208)𝑈𝑎 ∈ U 𝑈𝑎 = 𝑀+𝑟 , i.e., U is a cover of 𝑀+𝑟 .
• Local Closest-ness: if x ∈ 𝑈𝑎, for 𝑏 satisfies 𝑎 ⊂ 𝑏 or 𝑏 ⊂ 𝑎,

we have 𝑑𝑖𝑠 (x, 𝑎) < 𝑑𝑖𝑠 (x, 𝑏).

The covering property ensures the geometry we defined reflects
the offset geometry 𝑀+𝑟 . However, it added more information to
𝑀+𝑟 . The local exclusiveness ensures that each block 𝑈𝑎 can be a
unique indicator of layers of intersections of the offset geometry,
such as the overlapped part shown in Figure 4c.

It is worth noting that the block of a saddle vertex is an empty
set, i.e., it will not contact with any other point. This is acceptable
because if a point’s distance to such a vertex is less than 𝑟 , there
must be at least one neighbor face or edge that is contacting with
such a point.

3.4 Penetration Depth

Akin to Chen et al. [2023], each layer of intersection requires a
separate contact force to resolve. Naturally, we want to push the
intersecting point along the normal direction to the boundary.

From the definition of blocks provided in Equation 7, we can see
that if x ∈ 𝑈𝑎, the distance to the surface of 𝑈𝑎 along the normal
direction is:

𝑑𝑝 = 𝑟 − ||x − c(x, 𝑎)|| = 𝑟 − 𝑑𝑖𝑠 (x, 𝑎),

(13)

we refer to 𝑑𝑝 as the penetration depth of point x in 𝑈𝑎. 𝑑𝑝 is a
function of the vertex position p and c(p, 𝑎). Therefore, the normal
contact potential derived from 𝑑𝑝 still accords with the formulation
Equation 1.

3.5 Offset Geometry for Edge-edge Contact

We have defined the offset manifold for vertex-facet contact. Now
we can define a new geometry by offsetting all the edges to support
edge-edge contact. We extract all the vertices and edges of 𝑀 to
construct a new geometry 𝑀𝑒 , which we refer to as the edge-only
manifold. 𝑀𝑒 will be a 1-dimensional manifold which only contains
𝑀’s wireframes.

In 𝑀𝑒 , the Gauss map of an edge 𝑒 is a circle perpendicular to 𝑒
(see Figure 8a), and its corresponding block forms a cylinder with
𝑒 being its axis. The Gauss map of a vertex 𝑣 is a sphere cut by 2
planes perpendicular to 𝑣’s two neighbor edges, as illustrated in
Figure 8b, with its block being shaped correspondingly (Figure 8d).

The constructive definition of the edge block of 𝑀𝑒 is:

𝑒 ={x ∈ R𝑁 |
𝑈 𝐸

𝑑𝑖𝑠 (x, 𝑒) ≤ 𝑟
(x − x(𝑣𝑒,1))(x(𝑣𝑒,2) − x(𝑣𝑒,1)) > 0,
(x − x(𝑣𝑒,2))(x(𝑣𝑒,1) − x(𝑣𝑒,2)) > 0}

(14)

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

(a) Edge Gauss Map

(b) Vertex Gauss Map

(c) Edge Block

(d) Vertex Block

Fig. 8. Illustration of Gauss Maps and blocks in the edge-only mani-
fold 𝑀𝑒 of an edge and a vertex, respectively. The black dot represents
the vertex in the vertex block diagram, and the dashed lines indicate
open boundaries in the edge block diagram.

Similarly, the constructive definition of the vertex block of 𝑀𝑒 is:
𝑈 𝑒
𝑣 = {x ∈ R𝑁 | ||x − x𝑣 || ≤ 𝑟, (x − x𝑣)(x𝑣 − x𝑣′ ) ≥ 0, ∀𝑣 ′ ∈ V𝑣 }
(15)
The difference between edge-edge contact and vertex-facet con-
tact is, that the force is applied on an edge instead of a single vertex.
Similarly, we can define the normal contact potential for the edge-
edge contact:

𝐸𝑒𝑒
𝑛 (𝑀) =

∑︁

𝑔(𝑑𝑖𝑠 (𝑒, 𝑒′), 𝑟 )

(16)

𝑒,𝑒 ′ ∈ E𝐶

OGC
where EOGC is the set of all the actively contacting edge-edge pairs:
EOGC ={{𝑒1, 𝑒2} |
𝑒1, 𝑒2 ∈ E,
𝑒1 ∩ 𝑒2 = ∅,
∃𝑎 ⊂ 𝑒𝑖, c(𝑒𝑖, 𝑒 𝑗 ) ∈ 𝑈𝑎holds for 𝑖 = 1, 𝑗 = 2 and 𝑖 = 2, 𝑗 = 1}
(17)

where c(𝑒𝑖, 𝑒 𝑗 ) is 𝑒𝑖 ’s closest point to 𝑒 𝑗 .

The contact force between two edges is applied on two points:
c(𝑒, 𝑒′) and c(𝑒′, 𝑒). Li et al. [2020] provided a way to smoothly filter
out the parallel edge contact to avoid instability. Here we apply a
similar procedure to our edge-edge contact force.

3.6 A 𝐶2 Continuous 2-Stage Activation Function

The quadratic activation function is widely used because of its
simplicity and non-stiff nature. However, it has a drawback: the
contact normal force does not become infinite as two primitives
approach each other. This would result in penetration when the
large forces are pushing primitives towards each other.

To make sure the contact force will eventually get strong enough
to overcome all other forces to successfully separate contacting
primitives, the barrier activation function [Li et al. 2020] became
a popular choice. Their barrier function is a logarithmic function,

multiplied by a quadratic function to make sure it is 𝐶2 continuous
at the point where the contact force disappears.

We propose a novel 2-stage activation function, which possesses

the advantage of both of those energies:

𝑔(𝑑, 𝑟 ) =

(cid:40) 𝑘𝑐

2 (𝑟 − 𝑑)2
−𝑘′
𝑐𝑙𝑜𝑔(𝑑) + 𝑏

if 𝜏 ≤ 𝑑 ≤ 𝑟
if 0 < 𝑑 < 𝜏

(18)

where 𝑘𝑐 and 𝑘′
𝑐 are 2 stiffness factors of the 2 stages, 𝜏 is a parameter
determining where to stitch between those 2 stages. To make the
function 𝐶1 continuous at 𝑑 = 𝜏, 𝑘′

𝑐 and 𝑏 need to satisfy:

𝑘′
𝑐 = 𝜏𝑘𝑐 (𝜏 − 𝑟 )2

(𝑟 − 𝜏)2 + 𝑘′

𝑐𝑙𝑜𝑔(𝜏)

𝑏 =

𝑘𝑐
2

(19)

(20)

This leaves us only one configurable parameter 𝑘𝑐 , from which 𝑘′
𝑐
and 𝑏 can be computed accordingly. We further let 𝜏 = 𝑟
2 to make it
𝐶2 continuous.

This is a combination of a pure quadratic function and a pure
logarithmic function. With the 𝑘𝑐 properly set, most of the contacts
will be handled in the quadratic stage, benefiting from the faster
convergence of the quadratic function. In the second stage, since it is
a pure logarithmic function, it is still less stiff than IPC’s formulation
Li et al. [2020].

Combining this activation function with our contact sets FOGC
and EOGC, we have obtained a normal contact energy that is 𝐶2
continuous on most part (see the explanation in the Limitation
Section) of U. Additionally, the normal contact force derived from
this normal contact energy is always orthogonal to the primitive it
acts upon.

Friction. With the properly designed normal contact force,
3.6.1
we can compute the friction force using the friction coefficient to
compute the friction force. We use the lagged formulation of friction
provided by Li et al. [2020], with the modification proposed by Chen
et al. [2024b] to improve the stability.

3.7 Penetration-Free Simulation

We employ the technique provided in [Wu et al. 2020] to guarantee
penetration-free simulation. This technique relies on computing a
conservative bound for each vertex 𝑣:
𝑏𝑣 = 𝛾𝑝 min(𝑑min,𝑣, 𝑑𝐸

(21)
where 0 < 𝛾𝑝 < 0.5 is a relaxation parameter and 𝑑min,𝑣 is 𝑣’s
minimal distance to all the facets that do not include 𝑣:
𝑑min,𝑣 = min
𝑡 ∈ T,𝑣∉𝑡

min,𝑣, 𝑑𝑇

𝑑𝑖𝑠 (x𝑣, 𝑡),

min,𝑣),

(22)

and 𝑑𝐸
distances to all other edges:

min,𝑣 is the minimal value of 𝑣’s neighbor edges’ minimal

𝑑𝐸
min,𝑣 = min
𝑒 ∈ E𝑣

𝑑min,𝑒,

𝑑min,𝑒 =

min
𝑒 ′ ∈ E,𝑒∩𝑒 ′=∅

𝑑𝑖𝑠 (𝑒, 𝑒′),

(23)

(24)

and 𝑑𝑇
distances to all other vertices:

min,𝑣 is the minimal value of 𝑣’s neighbor facets’ minimal

𝑑𝑇
min,𝑣 = min
𝑡 ∈ T𝑣

𝑑min,𝑡 ,

(25)

Offset Geometric Contact

•

9

𝑑min,𝑡 = min

𝑑𝑖𝑠 (𝑣 ′, 𝑡),

(26)

𝑣′ ∈ V,𝑣′⊄𝑡
where E𝑣 and T𝑣 represents 𝑣’s neighbor edges and facets respec-
tively.

If the model starts in an intersection-free state 𝑋 prev, it will re-

main intersection-free in state if each x𝑣 satisfies:
||x𝑣 − xprev

|| ≤ 𝑏𝑣, ∀𝑣 ∈ V.

𝑣

(27)

𝑣

Starting with a penetration-free state 𝑋 prev, our method computes
𝑏𝑣 and records xprev
for each 𝑣. Then after some solver iterations,
each vertex will reach a new position x𝑣. Our method checks each
vertex individually to see if it satisfies the condition in Equation 27. If
a vertex does not satisfy the condition, its displacement is truncated
to stay within the conservative bound. 𝑏𝑣. Subsequently, our method
recomputes 𝑏𝑣 and records x as the penetration-free starting state,
repeating this process iteratively.

Since the conservative bound and 𝑋 prev can be updated as needed
during the solver’s iterations, they do not restricted each vertex’s
total displacement within a time step, only limiting the displacement
within each individual iteration. A vertex near an obstacle may
be constrained in initial iterations, but the bound-update will be
triggered once it reaches the bound. The new bound becomes larger
since the repulsive force pushes it away from the obstacle. This
procedure will repeat until convergence. As shown in Figure 17, the
solver converges under these bounds without introducing additional
artifacts.

3.8 Comparing to IPC

Our method can be viewed as a trust region based method for a
constrained optimization problem, where the constraints are the
penetration-free constraints. The spherical region we compute for
each vertex is the trust region we formulate to enforce the con-
straints. In contrast, IPC [Li et al. 2020] employs a CCD-aware line
search technique to achieve penetration free-state, which requires
truncating the step.

The CCD-aware line search technique maintains a penetration-
free state by applying CCD after every iteration of the optimization.
It truncates the optimization step of the physics solver at where the
first collision happens, thereby preventing penetration. However,
this also means a local collision stops the progress of all other points,
even if those points are still far from intersecting. This is illustrated
in Figure 9b, where the whole step is stopped by the vertex in
the middle which is closest to the obstacle. Each iteration can be
computationally expensive, and truncating the global optimization
step entirely often results in only a small fraction of the step being
utilized. This approach overlooks the fact that most parts of the
model could still make significant progress along the optimization
direction, leading to wasted computation and slower convergence.
This issue becomes particularly evident when parts of the model
are in close proximity to one another.

In our formulation, 𝑏𝑣 is different for each 𝑣 ∈ V, as illustrated
in Figure 9c. The value of 𝑏𝑣 is smaller in regions that are actively
in contact and larger in regions that are far from others. As a result,
each 𝑏𝑣 has only a local impact. Even when certain parts of the
model are close to each other, such as the vertices in the middle of
Figure 9c, the conservative bounds of other regions, like the vertices

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

10

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

(a) Previous Position and a Full
Optimization Step

(b) Iteration Using IPC’s Scheme

(c) Conservative Bounds

(d) Iteration Using by Our Scheme

Fig. 9. Comparing different schemes to preserve penetration-free state
in a single solver iteration. The black line represents the shape
of 𝑀, the dashed gray line represents the destination position after
taking a full step given by the optimizer in that iteration, the red
circle represents an obstacle, and the green dots represent vertices of
𝑀. (a) The penetration-free position 𝑋 prev in the previous iteration,
and a position 𝑋 + Δ𝑋 after taking a full optimization step, which
presents penetration; (b) the penetration-free optimization step given
using IPC’s CCD-aware line search scheme; (c) illustration of out
conservative bounds 𝑏𝑖 which vary at each vertex; (d) the penetration-
free optimization step given by our scheme.

on the sides, remain unaffected. These unaffected regions can still
utilize relatively large step sizes.

Our contact force formulation allows for a significantly larger
contact radius compared to IPC. As a result, even primitives that
are actively in contact can maintain a relatively large distance. This,
in turn, enables our method to take bigger steps per iteration and
achieve fast convergence, despite employing conservative bound
truncation. In contrast, simply combining our trust-region-based
optimization scheme with IPC’s contact energy will not work, be-
cause a small contact radius as IPC uses will result in a near-zero
conservative bound..

Furthermore, as we will explain in the next section, there is no
need to use a separate function to compute 𝑏𝑣. Instead, this can
be seamlessly integrated into the contact detection process with
negligible overhead. Additionally, the computation of 𝑏𝑣 is fully
parallel, and does not require CPU-GPU synchronization when
implemented on GPU. This approach offers a significant advantage
over the CCD-based line search employed by IPC, which requires
multiple computationally expensive continuous collision detections
and synchronizations in a single iteration.

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

3.9 Offset Geometry for Mesh with Different

Dimensionalities

It is straightforward to apply OGC to the surface of a volumetric
object to ensure penetration-free contact simulation. However, when
simulating volumetric objects, users can choose either to enforce
penetration constraints for robustness or to omit them for efficiency,
as volumetric intersections can be resolved after occurrence (Chen
et al. [2023]). Therefore, for volumetric mesh simulations, we can
skip conservative bound culling to accelerate convergence. Here we
propose a faster method specially tailored for the volumetric mesh
simulations.

For a volumetric mesh 𝑀, the offset operation should be applied
to its surface 𝜕𝑀 to obtain an intersection-aware offset geometry
U (𝜕𝑀). Note that in this case, we only offset the geometry out-
ward, in the direction of the surface normal. The penetration depth
computed from U (𝜕𝑀) is compatible with the penetration depth
provided by Chen et al. [2023].

We use pure quadratic contact energy in volumetric simulation,
e.g., only using the first stage of Equation 18. At the beginning of
each step, the simulator performs DCD (discrete collision detection)
for each vertex to determine whether they have intersected 𝑀. If
penetration is detected, the simulator computes the penetration
depth 𝑑𝑝 using the method proposed by Chen et al. [2023]. This
penetration depth needs to be adjusted to 𝑑𝑝 + 𝑟 to match the pen-
etration depth of the offset geometry. If no intersection is found,
the simulator performs another DCD to detect its intersection with
U (𝜕𝑀) and computes the penetration depth in U (𝜕𝑀) . This en-
sures that the penetration depths in 𝑀 and U (𝜕𝑀) are consistent,
resulting in consistent contact forces from both the inside and out-
side of the mesh. The contact force is greater than 0 at 𝜕𝑀 due to the
offset layer. With properly adjusted contact stiffness, most contact
will still occur outside the mesh, maintaining an intersection-free
state for most parts of the mesh. We use this scheme to handle the
cloth-body contact in our cloth simulation experiments.

For 1D meshes immersed in 3D space, such as those used in hair
and yarn-level simulations, they can be treated in the same way
as the edge-only manifold proposed in Section 3.5. Specifically, we
employ this contact model to generate the yarn simulation results
presented in Figure 15 and Figure 16.

4 ALGORITHM

We have defined the contact force and energy for the vertex-facet
contact and the edge-edge contact. Now we propose the algorithms
to practically detect those contacts. Since the intersection-aware off-
set geometry is composed of many blocks, a trivial implementation
will be building a BVH (Bounding Volume Hierarchy) of all those
blocks, and looping through all the vertices and edges to detect
intersections with those blocks.

However, these blocks correspond to different types of faces,
including vertices, edges, and facets. Building a BVH that contains all
these blocks would result in an excessively large structure. Instead,
we present a method that only requires building a BVH for the
faces with the highest dimensionality: facets for vertex-facet contact
detection and edges for edge-edge contact detection. Note that such
a BVH is constructed based on the bounding boxes of original faces,

not the offset ones. Additionally, those algorithms are capable of
computing 𝑑min,𝑣, 𝑑𝐸
min,𝑣 simultaneously with the contact
detection.

min,𝑣, and 𝑑𝑇

4.1 Vertex Facet Contact

Algorithm 1: vertexFacetContactDetection
Input: 𝑣: a vertex, 𝑟 : contact radius, 𝑟𝑞: query radius
Output: FOGC (𝑣): set of faces that are in contact with 𝑣;
VOGC (𝑡): set of vertices that are in contact with 𝑡;
𝑑min,𝑣: the minimal distance from 𝑣 to another faces;
𝑑min,𝑡 : the minimal distance from 𝑡 to all other vertices.

1 𝑑min,𝑣 = 𝑟𝑞

// sphere query on the facet BVH with center x(𝑣) and radius 𝑟𝑞

3

2 for each 𝑡 ∈ T s.t. 𝑑𝑖𝑠 (𝑣, 𝑡) < 𝑟𝑞 do
// avoid contact with adjacent facet
if 𝑣 ⊂ 𝑡 then continue
𝑑 = 𝑑𝑖𝑠 (𝑣, 𝑡)
𝑑min,𝑣 = 𝑚𝑖𝑛(𝑑, 𝑑min,𝑣)
// multiple vertex query threads may access the same 𝑑𝑚𝑖𝑛,𝑓
simultaneously, thus this must be an atomic min operation

4

5

𝑑min,𝑡 = min(𝑑, 𝑑min,𝑡 )
if 𝑑 < 𝑟 then

6

7

8

9

10

11

12

13

14

15

16

17

18

19

𝑎 = closestFaceFacetToVertex(𝑣, 𝑡)
// avoid duplicated contact with 𝑎 detected from a neighbor facet
if 𝑎 ∈ FOGC (𝑣) then continue
if 𝑎 ∈ V then
// Equation 8
if checkVertexFeasibleRegion(x(𝑣), 𝑎) then

FOGC (𝑣) = FOGC (𝑣) ∪ {𝑎}
VOGC (𝑡) = VOGC (𝑡) ∪ {𝑣 }

else if 𝑎 ∈ E then
// Equation 9
if checkEdgeFeasibleRegion(x(𝑣), 𝑎) then

FOGC (𝑣) = FOGC (𝑣) ∪ {𝑎}
VOGC (𝑡) = VOGC (𝑡) ∪ {𝑣 }

else

// 𝑣 must be in the feasible region in this case
FOGC (𝑣) = FOGC (𝑣) ∪ {𝑡 }
VOGC (𝑡) = VOGC (𝑡) ∪ {𝑣 }

20
21 end
22 return FOGC (x𝑣), 𝑑min,𝑣

The algorithm for detecting vertex-facet contact is provided in
Algorithm 1. As previously mentioned, we only maintain a BVH for
all the facets. To detect vertex-facet contact, we do a point query
with center x(𝑣) and radius 𝑟𝑞 for each vertex 𝑣 ∈ V. 𝑟𝑞 is a custom
parameter satisfies 𝑟𝑞 ≥ 𝑟 .

For each facet 𝑓 within the query radius 𝑟𝑞, the algorithm com-
putes its closest point to 𝑣, the face 𝑎 on the facet where the closest
point is located, and the distance 𝑑 = 𝑑𝑖𝑠 (𝑣, 𝑡) (line 4,5,8). Note that 𝑎
can be either a vertex, or an edge, or 𝑡 ◦. Then it updates 𝑣’s minimal
distance to facets, 𝑑min,𝑣. We also update 𝑓 ’s minimal distance to

Offset Geometric Contact

•

11

vertices in parallel, 𝑑min,𝑡 , using an atomic min operation. This is
to avoid a race condition since multiple vertex query threads can
access the same 𝑑min,𝑡 simultaneously.

Since all the vertices whose distance to 𝑡 is less than 𝑟𝑞 will visit
𝑡, this ensures that we are computing the correct 𝑑min,𝑡 . Both 𝑑min,𝑣
and 𝑑𝑚𝑖𝑛,𝑓 are initialized as 𝑟𝑞, because the query does not look
beyond that distance. This means that even if there are no active
contact pairs detected, 𝑑min,𝑣 and 𝑑𝑚𝑖𝑛,𝑓 are still upper-bounded by
𝑟𝑞, because we do not know if there is a facet whose distance to
𝑣 is marginally larger than 𝑟𝑞. That is why we make 𝑟𝑞 a separate
parameter. Making 𝑟𝑞 larger than 𝑟 will not detect more contacts, but
it can potentially improve the conservative bound for each vertex,
thereby enhancing convergence. In practice, we found an 𝑟𝑞 of 𝑟
plus the inertial displacement magnitude to strike a good balance
between query performance and bound size.

The next step will be determining whether 𝑎 is in contact with 𝑣,
i.e., whether 𝑣 ∈ 𝑈𝑎. Note that when 𝑎 is not a facet, it is shared by
multiple neighboring facets. In this case, multiple facets can return
the same closest face 𝑎. To avoid duplicated contacts, we check
whether 𝑎 already exists in the contacting face set F𝑂𝐺𝐶 (x𝑣). If
𝑎 ∉ F𝑂𝐺𝐶 (𝑣), we proceed to check 𝑣 ∈ 𝑈𝑎 using 𝑈𝑎’s constructive
definition (Equation 8, Equation 9). Note that if the closest point is
located in the interior of 𝑡, i.e., 𝑎 = 𝑡, 𝑣 ∈ 𝑈𝑎 is guaranteed. Therefore,
no feasible region check is required in this case. For the convenience
of contact force computation, we also maintain a list VOGC (𝑡) for
each 𝑡 ∈ T , which is the set of vertices that are in contact with 𝑡.
After putting a face into F𝑂𝐺𝐶 (𝑣), we also put 𝑣 into VOGC (𝑡) of
the corresponding facet using atomic operation.

According to Equation 13, if 𝑎 ∈ 𝑡 contacts with 𝑣, it must be the
closest face on 𝑡 to 𝑣. The local exclusive property guarantees that
𝑣 can only be in contact with at most one face on 𝑡. If 𝑣 contacts
with 𝑎 ∈ 𝑡, it will not contact all other faces of 𝑡. Since 𝑣 will visit
all the facets whose distance to 𝑣 is less than 𝑟 , this guarantees that
Algorithm 1 will not miss or duplicate any vertex-facet contact.

4.2 Edge Edge Contact

Similarly, we give the algorithm that detects edge-edge contact in
Algorithm 2. Similar to Algorithm 1, it works on the BVH of all the
edges, and applies a sphere query centered at x𝑚 with radius 𝑟𝑞 + 𝑙
2
for each edge, where x𝑚 and 𝑙 are the midpoint and length of that
edge, respectively. Each query also computes the 𝑑min,𝑒 . Since every
edge has its query thread, no automatic operation is needed here.
Note that since each edge detects its own contacts, each edge-edge
contact will be automatically detected exactly twice: one from each
side.

4.3 Simulation Pipeline

Now that we have provided the contact energy and the algorithms
to detect those contacts, the next step would be integrating it into an
actual simulation pipeline. Theoretically, the contact force we for-
mulated can be used in a variety of time integrators, including both
explicit and implicit ones. Here we provide an algorithm combining
Offset Geometric Conact with backward Euler in Algorithm 3.

There are 3 major stages of the simulation pipeline: contact detec-
tion (line 4 ∼ 19), simulation solve (line 20 ∼ 22), and conservation

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

12

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

Algorithm 2: edgeEdgeContactDetection
Input: 𝑒: a edge, 𝑟 : contact radius, 𝑟𝑞: query radius
Output: EOGC (𝑒): set of faces contacting 𝑒;
𝑑min,𝑒 : the minimal distance from 𝑒 to all other edges.

1 𝑑min,𝑒 = 𝑟𝑞
2 x𝑚 = midpoint of 𝑒
3 𝑙 = length of 𝑒

Algorithm 3: Simulation Step with Offset Geometry Contact
Input: 𝑋 𝑡 ∈ R𝐾 ×3: stacked positions of vertices from previous step;
v𝑡 ∈ R𝐾 ×3: stacked velocities of vertices from previous step;
aext: external acceleration;
𝛾 : a parameter controls when to do a new collision detection;
𝑀 = { V, E, T };
𝑟 : contact radius, 𝑟𝑞: query radius
Output: 𝑋 ∈ R𝐾 ×3: stacked positions of vertices for current step

// sphere query on the facet BVH with center x𝑚 and radius 𝑟𝑞 + 𝑙
2

4 for each 𝑒′ s.t. 𝑑𝑖𝑠 (𝑒, 𝑒′) < 𝑟𝑞 + 𝑙

2 do

// avoid contact with adjacent edge
if 𝑒 ∩ 𝑒′ ≠ ∅ then continue
𝑑 = 𝑑𝑖𝑠 (𝑒, 𝑒′)
𝑑min,𝑒 = 𝑚𝑖𝑛(𝑑, 𝑑min,𝑒 )
if 𝑑 < 𝑟 then

x𝑐 = C(𝑒, 𝑒′)
𝑎 = closestFaceEdgetToEdge(𝑒, 𝑒′)
// avoid duplicated contact with 𝑎 detected from a neighbor facet
if {𝑒, 𝑎} ∈ EOGC (𝑒) then continue
if 𝑎 ∈ V then
// Equation 15
if checkVertexFeasibleRegionEdgeOffset(x𝑐, 𝑎)
then

EOGC (𝑒) = EOGC (𝑒) ∪ {𝑒, 𝑎}

else

// 𝑣 must be in the feasible region in this case
EOGC (𝑒) = EOGC (𝑒) ∪ {𝑒, 𝑒′}

5

6

7

8

9

10

11

12

13

14

15

16

return EOGC (𝑒), 𝑑min,𝑒

17
18 end

bound truncation (line 23 ∼ 30). We will introduce each stage corre-
spondingly.

4.3.1 Contact Detection. In the contact detection stage, the simula-
tor will apply the previously provided contact detection algorithms
to the model. Note that before we apply the vertex-facet contact
detection, we need to initialize all the 𝑑min,𝑓 to their upper-bound
𝑟𝑞 (line 6, 7). Then we apply all the vertex-facet contact and edge-
edge contact detections in parallel, which computes the contacting
faces and each face’s minimal distance from other faces. At last, the
simulator computes the conservative bounds 𝑏𝑣 for all the vertices
based on that information (line 17 ∼ 19).

Simulation Solve. The first step in simulation solving is to
4.3.2
apply an initialization that avoids penetration. A trivial approach is
to use the positions from the previous step, but a better initialization
can improve convergence and reduce damping. Since any guess
within conservative bounds will be penetration-free, we can choose
an arbitrary initialization scheme and truncate it to stay within
these bounds, ensuring a penetration-free start:

xinit∗
𝑣

=

xinit
𝑣
𝑣 −x𝑡
xinit
𝑣
𝑣 −x𝑡
| |xinit
𝑣 | |





𝑏𝑣 + x𝑡
𝑣

if ||xinit
if ||xinit

𝑣 − x𝑡
𝑣 − x𝑡

𝑣 || ≤ 𝑏𝑣
𝑣 || > 𝑏𝑣

(28)

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

1 collisionDetectionRequired = true
2 𝑋 = 𝑋 𝑡
3 𝑌 = 𝑋 𝑡 + Δ𝑡 v𝑡 + Δ𝑡 2aext
4 for each 𝑖 in 1, 2, . . . , 𝑛iter do
5

if collisionDetectionRequired then

// Initialize 𝑑min,𝑡 to their upper-bound
parallel for each 𝑡 ∈ T do

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

𝑑min,𝑡 = 𝑟𝑞

end
parallel for each 𝑣 ∈ V do
FOGC (𝑣), 𝑑min,𝑣 =
vertexFacetContactDetection(𝑣, 𝑟, 𝑟𝑞 )

end
parallel for each 𝑒 ∈ E do
EOGC (𝑒 ), 𝑑min,𝑒 =
edgeEdgeContactDetection(𝑒, 𝑟, 𝑟𝑞 )

end
𝑋 prev = 𝑋
collisionDetectionRequired=false

parallel for each 𝑣 ∈ V do

// Equation 21
𝑏𝑣 = computeConservative(𝑣)

end
if 𝑖 == 1 then

𝑋 = applyInitialGuess(𝑋 𝑡 , v𝑡 , aext )

𝑋 =
simulationIteration( { FOGC }, { VOGC }, { EOGC }, 𝑋 , 𝑋 𝑡 , 𝑌 , v𝑡 , aext, 𝑀)

numVerticesExceedBound = 0
// Truncated the vertex displacements to be within 𝑏𝑣
parallel for each 𝑣 ∈ V do
if | |x𝑣 − xprev
𝑣
x𝑣 = x𝑣 −x
||x𝑣 −x
// Atomic increment
numVerticesExceedBound++

| | > 𝑏𝑣 then
prev
+ xprev
𝑣
𝑣
prev
𝑣

||

end
if numVerticesExceedBound >= 𝛾𝑒 𝐾 then

// If a certain amount of vertices move out of their conservative

bounds, do a new collision detection
collisionDetectionRequired = true

// Optional Convergence Evaluation
if
evaluateConvergence( { FOGC }, { VOGC }, { EOGC }, 𝑋 , 𝑋 𝑡 , v𝑡 , aext, 𝑀 )
then

break

32
33 end
34 return 𝑋

Algorithm 4: VBD Iteration with Contact
Input: 𝑋 : the initialization value; 𝑋 𝑡 : the positions of the previous

step; 𝑌 : the inertia;

Output: This step’s position x𝑡 +1 and velocity v𝑡 +1.

1 for each color 𝑐 do

// Block-level parallelization
parallel for each vertex 𝑣 in color 𝑐 do
f𝑣 = − 𝑚𝑣
ℎ2 (x𝑣 − y𝑣 ), H𝑣 =
// Thread-level parallelization
parallel for each 𝑡 ∈ T𝑣 do

𝑚𝑣
ℎ2 I

// Variables in shared memory
𝜕2𝐸𝑡
f𝑣,𝑡 = − 𝜕𝐸𝑡
𝜕x𝑣 , H𝑣,𝑡 =
𝜕x𝑣 𝜕x𝑣

end
// Local reduction sums
f𝑣+= (cid:205)𝑡 ∈T𝑣 f𝑣,𝑡 , H𝑣+= (cid:205)𝑡 ∈T𝑣 H𝑣,𝑡
// Thread-level parallelization
parallel for each 𝑒 ∈ E𝑣 do
// Variables in shared memory
𝜕2𝐸𝑒
f𝑣,𝑒 = − 𝜕𝐸𝑒
𝜕x𝑣 , H𝑣,𝑒 =
𝜕x𝑣 𝜕x𝑣

end
// Local reduction sums
f𝑣+= (cid:205)𝑒 ∈E𝑣 f𝑣,𝑒 , H𝑣+= (cid:205)𝑡 ∈E𝑣 H𝑣,𝑒
// Accumulate the force and Hessian of the vertex-facets contact of

the vertex side

// Thread-level parallelization
parallel for each 𝑎 ∈ FOGC (𝑣) do
// Variables in shared memory
𝑣,𝑓
𝜕𝐸
𝑐
𝜕x𝑣

f𝑣,𝑎 = −

, H𝑣,𝑎 =

(𝑣,𝑎)

𝑣,𝑓
𝜕2𝐸
𝑐

(𝑣,𝑎)

𝜕x𝑣 𝜕x𝑣

end
// Local reduction sums
f𝑣+= (cid:205)𝑎∈FOGC (𝑣) f𝑣,𝑎, H𝑣+= (cid:205)𝑎∈FOGC (𝑣) H𝑣,𝑎
// Accumulate the force and Hessian of the vertex-facets contact of

the neighbor facets
for each 𝑡 ∈ T𝑣 do

// Thread-level parallelization
parallel for each 𝑣′ ∈ VOGC (𝑡 ) do
// Variables in shared memory
𝑣,𝑓
𝜕𝐸
𝑐

(𝑣′,𝑡 )

f𝑡,𝑣′ = −

𝜕x𝑣

, H𝑡,𝑣′ =

𝑣,𝑓
𝜕2𝐸
𝑐
𝜕x𝑣 𝜕x𝑣

(𝑣′,𝑡 )

end
// Local reduction sums
f𝑣+= (cid:205)𝑣′ ∈VOGC (𝑡 ) f𝑡,𝑣′ , H𝑣+= (cid:205)𝑣′ ∈VOGC (𝑡 ) H𝑡,𝑣′ ;

end
// Accumulate the force and Hessian of the edge-edge contact of

neighbor edges
for each 𝑒 ∈ E𝑣 do

// Thread-level parallelization
parallel for each 𝑒′ ∈ EOGC (𝑒 ) do
// Variables in shared memory

f𝑒,𝑒′ = −

𝜕𝐸𝑒,𝑒
𝑐

(𝑒,𝑒′ )

𝜕x𝑣

, H𝑒,𝑒′ =

𝜕2𝐸𝑒,𝑒
𝑐

(𝑒,𝑒′ )

𝜕x𝑣 𝜕x𝑣

end
// Local reduction sums
f𝑣+= (cid:205)𝑒′ ∈EOGC (𝑒 ) f𝑒,𝑒′ , H𝑣+= (cid:205)𝑒′ ∈EOGC (𝑒 ) H𝑒,𝑒′ ;

end
x𝑣 ← x𝑣 + H−1

𝑣 f𝑣

2

3

4

5

6

7

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

end

29
30 end
31 return 𝑋

Offset Geometric Contact

•

13

are the initialization post and pre truncation,

where xinit∗
𝑣
respectively, x𝑡

and xinit

𝑣

𝑣 is 𝑣’s position at the last step.

The second step is solving the non-linear equation of the back-
ward Euler time integration. OGC is compatible with various solvers,
such as Newton’s method, gradient descent, and block coordinate
descent, provided they work with the energy formulation of OGC.
These solvers can be seen as functions that yield a displacement
from the previous position to reduce the energy. To prevent pene-
tration, we need to post-process the displacements by truncating
them within the conservative bounds.

Here we present an efficient GPU implementation of a VBD [Chen
et al. 2024b] solver, as shown in Algorithm 4. In this algorithm, 𝑚𝑣
denotes the mass of vertex 𝑣, 𝐸𝑡 the elastic energy of facet 𝑡, 𝐸𝑒 the
bending energy of edge 𝑒, and 𝐸𝑣,𝑓
represent the contact
𝑐
energies (including both normal and frictional components) for
vertex-facet and edge-edge contacts, respectively. We employ a two-
level parallelism scheme similar to that in [Chen et al. 2024b], except
we use thread-level parallelism to accumulate contact forces and
Hessians for each vertex, as well as for its neighboring facets and
edges.

and 𝐸𝑒,𝑒
𝑐

4.3.3 Conservative Bound Truncation. According to Equation 27,
starting from a penetration-free state 𝑋 prev, as long as the displace-
ment of each vertex satisfies ||Δx𝑣 || < 𝑏𝑣, it is guaranteed that
the model will not create any penetration. Note that Δx𝑣 may not
be the displacement of a single iteration of simulation but can be
the accumulated displacement from multiple iterations. Therefore,
collision detection is not needed in every iteration to guarantee a
penetration-free state. Only when some vertices have exceeded their
conservative bounds, new collision detection is needed to refresh
the conservative bounds and recalculate the contacts.

This property is particularly beneficial for first-order or locally
second-order methods, such as gradient descent or vertex block
descent, since these methods create relatively small displacements
at each iteration, and each iteration is very fast. For these methods,
new collision detection is typically needed only after a fair amount
of iterations.

During the collision detection stage, the simulator records the
position where the contact detection is conducted as 𝑋 prev (line
15). After each simulation iteration, the simulator computes the
displacement of each vertex from 𝑋 prev, and truncates the displace-
ment to be within the bounds (line 25 ∼ 27). Instead of redoing
contact detection every time one vertex moves out of its bound, a
threshold 𝛾𝑒 is used to control when to apply a new contact detec-
tion. A new contact detection is performed only after the number of
vertices moving out of their bounds exceeds 𝛾𝑒𝐾 (line 29, 30). Before
this threshold is reached, those vertices can be truncated multiple
times and cannot move any further, though they can still adjust the
direction of their displacement.

5 RESULTS

We implemented our algorithm on both CPU and GPU platforms.
The CPU implementation, written in C++, was executed on an
AMD Ryzen 7950X with 64 GB of memory. We implement the GPU
version using NVIDIA Warp [Macklin 2022], and run it on a NVIDIA
RTX 4090. For the simulation parameters, we set 𝛾𝑝 = 0.45 and

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

14

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

Table 1. Performance results and simulation

Experiment Name

50 Layers of Cloth (Figure 10)
Tightening a knot (Figure 11)
Twisting Cloth (Figure 12)
Cloth on Body (Figure 14)
Robot and T-shirt (Figure 14)
Yarn Stretch (Figure 15)
Yarn Twist (Figure 16)
3 Layers of Cloth on Sphere (Figure 17)
1 Layer of Cloth on Sphere (Figure 18)
Twisting Volumetric Mat (Figure 19)

Primitives
1.96M
92K
19.6K

Number of
Vert.
1M
48K
10K
15.6K 29K
13.8K 27.4K
65K
65K
14.7K 28.6K
9.5K
4.9K
46.8K
15K

65K
65K

Contact &Fiction
𝜇𝑐, 𝜖𝑣
𝑘𝑐
0.2, 1e-2
1e5
0.4, 1e-2
1e5
0.2, 1e-2
1e5
0.5, 1e-2
1e5
0.5, 1e-2
1e5
0.1, 1e-3
2e-3
0.1, 1e-3
2e-3
0.5, 1e-2
1e4
0.5, 1e-2
1e4
0.2, 1e-2
1e5

𝑟 (𝑚𝑚)
2
2
2
2
2
1.5
1.5
2
5
2

Time per step (avg./max)

Simulation Parameters
Time Step (sec.)
1/1200
1/300
1/300
1/200
1/600
4e-4
4e-4
1/100
1/100
1/240

Iterations CPU VBD GPU VBD
6.3/11.5ms
0.21/0.55s
40
4.4/6.8ms
122/180ms
50
0.9/1.5ms
21/30ms
10
1.2/1.4ms
30/42ms
20
1.8/2.2ms
NA
10
0.23/0.30ms
NA
4
0.25/0.33ms
NA
4
NA
See figure
NA
1.2/3.2ms
62/75ms
40
5.5/8.5ms
NA
20

𝛾𝑒 = 0.01. We choose a relatively conservative value for 𝛾𝑝 to ensure
the conservative bounds remain conservative with floating point
rounding errors. Details of other experiment-specific parameters
and performance metrics are provided in Table 1. To ensure that all
our results are penetration-free, we perform an intersection analysis
after every frame and halt if any intersections occur. We plan to
open-source both the C++ and Warp versions of our code, ensuring
they are user-friendly and ready for out-of-the-box use.

5.1 Cloth Simulation

5.1.1 Large Scale Test. To evaluate the stability, efficiency, and
scalability of our method, we present a simulation of colliding 50
layers of cloth. Those clothes are dropped onto a cylinder, collide
with each other, and then slide to the ground. They form a pile on
the ground and eventually rest in contact. The dropping process is
visualized in Figure 10, and the final state of the simulation can be
seen in Figure 1. Despite the complicated contacts, the simulation
remains penetration-free the entire time.

Stress Tests. We present two experiments to demonstrate the
5.1.2
stability and performance of our method in scenarios involving
numerous complex self-collisions, extreme normal contact forces,
and frictions. Both of those experiments maintain penetration-free
states the entire time.

The first experiment, illustrated in Figure 11, simulates the for-
mation of a complex tight knot. We initialize the knot in a loose
form, then tighten it by pulling its two ends. As the knot tightens,
small sub-knots form and collide with each other. Eventually, these
small knots merge into a tight, multi-layered complex knot.

The second experiment, illustrated in Figure 12, involves twisting
a square-shaped cloth’s two ends for eight complete turns. This
example features extreme deformations, generating strong material
forces that compete with self-collisions. Our contact model effec-
tively handles these strong deformations with frictional contact.

5.1.3 Coupled Cloth Simulation. To test our method in a practical
cloth simulation scenario, we conduct the same cloth simulation
experiment as the one presented in the C-IPC paper [Li et al. 2021],
as shown in Figure 14. The cloth consists of 14 separate pieces
stitched together using stiff zero-length spring constraints, see Fig-
ure 13a. We filter out collisions between the stitched primitives
to ensure seamless stitching. For body-cloth contact, we use volu-
metric contact energy since the body motion is driven by skeleton

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

animation rather than simulation, making it challenging to prevent
penetration after updating the body’s position, see Figure 13b. As a
result, body-cloth penetration might occur during the simulation,
but cloth-cloth penetration is prevented. In the C-IPC paper, the
average computational time for a 0.04-second frame is reported as
24 seconds. In our tests, the same frame takes only 0.24 seconds on
the CPU and 9.6 milliseconds on the GPU.

We also showcase a scenario of a robot manipulating a T-shirt.
The robot’s trajectory is pre-computed and we use the same scheme
as in Figure 14 to handle the collision between the cloth and the
robot. This experiment runs in real time and stably simulates the
contacts between the robot and multiple layers of cloth.

5.2 Yarn Level Simulation

We perform further stress tests of our method with the use of yarn
level cloth simulations. Instead of simulating cloth as thin shells, we
individually simulate each constituent yarn thread as codimensional
rods. The behavior of the cloth is then the sum of contributions
from yarn bending, twisting, stretching, contacts, and friction. This
is traditionally difficult as even minor penetrations (pull-throughs)
can cause significant unraveling of the yarn.

We model rod bending, twisting, and stretching with the use of
Cosserat Rods similar to that proposed by Kugelstadt and Schömer
[2016]. To demonstrate the effect of our conservative bounds, we
implement a penalty-energy-based collision handling method and
compare it against our method. This method models contact as a
quadratic energy and always takes the full step given by Newton’s
method. In the figure, we label this method as "Newton".

In Figure 15, we pull and stretch the yarn cloth on two ends until
it is taut with tension. Using penalty-energy-based collisions, the
yarn threads phase through each other as they ultimately unravel
catastrophically. Our method successfully preserves the yarn ge-
ometry even under extreme tension. Despite the yarn being taut
enough to remain flat against gravity, no pull-through occurs.

In Figure 16, we clamp the square yarn cloth on two ends and twist
one end in five full rotations. Penalty-energy-based collisions fail
to prevent yarn penetration as the yarn threads crush and entangle
into a knot. In contrast, our method is able to successfully return
to the original state once the cloth is let go with no change to the
yarn structure.

In both examples, we use a yarn cloth that is 40cm by 40cm with
3mm thick yarn under normal gravity. The yarn threads have a

Offset Geometric Contact

•

15

Fig. 10. Fifty layers of cloth are dropped onto a cylinder, then slide to the ground. This simulation has 246K vertices and 475K triangles. We use
𝑟 = 3mm, a time step of 1/1200s, and 40 iterations per step. The average/maximum computation time per time step is 0.21/0.55s on the CPU and
6.3/11.5ms on the GPU.

Fig. 11. Tightening a complex knot with 48K vertices and 91K faces by pulling its two ends. At the end of the simulation, the mesh forms a
multi-layered, very tight knot. We use 𝑟 = 2mm, a time step of 1/300s, and 50 iterations per step. The average/maximum computation time per
time step is 122/180ms on the CPU and 6.3/11.5ms on the GPU.

Fig. 12. Twisting a square cloth for 8 circles, showcasing complicated self-collision with extreme contact force and friction. The model has 10K
vertices and 19.6K faces. We use 𝑟 = 2mm, a time step of 1/300s, and 10 iterations per step. The average/maximum computation time per time step
is 21/30ms on the CPU and 0.9/1.5ms on the GPU.

density of 1 gram per meter and a friction coefficient of 0.1. Both
examples can run about 1.8 times faster than real time on our setup.

5.3 Convergence

To evaluate OGC’s ability to converge with different solvers, we
plot the change in relative force residuals over iterations and com-
putation time in Figure 17. The relative force residual is defined
as:

𝑒 (𝑖 ) =

(𝑖 )
mean(||f
||)
𝑣
mean(||f ∗
𝑣 ||)

𝑣 is the initial force residual on vertex 𝑣 and f

where f ∗
residual on vertex after iteration 𝑖.

Both VBD and Newton’s method can reduce the mean force residu-
als to less than 1e-4, which is the lowest error achievable with single
precision. We run VBD for 500 iterations and Newton’s method for

(29)

(𝑖 )
𝑣

is the force

50 iterations. The spike in VBD’s curve corresponds to the applica-
tion of contact detection. Since we are not performing a line search
for VBD, the force residuals experience a brief spike after updating
the contact set through a new DCD. However, VBD quickly recovers
from this with just a few iterations and continues to reduce the
error.

In terms of convergence speed, Newton’s method converges faster
in terms of iterations, reaching numerical convergence at the 46th
iteration. However, since each iteration of Newton’s method is much
more computationally expensive and requires a line search to ensure
stability, it lags far behind VBD in terms of computational time. Col-
lision detection accounts for approximately 3% of the computational
time when using Newton’s method and 10% when using VBD. This
experiment demonstrates that the contact force defined by OGC

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

16

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

(a) Template

(b) Initial State

Fig. 13. Simulating a dress on a moving human body. The character is
driven by skeletal animation, with 12.8K vertices and 25.4K triangles.
The dress model consists of 14 separate pieces as shown in (a), with
15.7K vertices and 29.4K triangles. We use 𝑟 = 2mm, a time step of
1/200s, and 20 iterations per step. The average/maximum computation
time per time step is 30/42ms on the CPU and 1.2/1.4ms on the GPU.

can converge very efficiently with various solvers, using minimal
contact detections.

5.4 Quantitative Comparison to Incremental Potential

Contact

We compare Newton’s method based OGC and VBD based OGC
with IPC with incremental potential contact (IPC). The results are
visualized in Figure 18. We used the open-sourced implementation
of Codimensional-IPC (C-IPC) to generate results. We evaluate the
computational time and the total number of collisions at each step.
We can see that IPC uses significantly more collision detections
because IPC require more than two collision detections at each
iteration: one CCD to cull the global step size and one DCD per
energy evaluation in the line search process. This is not the case for
OGC-based methods, because OGC does not require any CCD, and
DCD is only necessary when points reach their conservative bounds,
which does not happen at every iteration, especially in the later
stages of the optimization process. In these later stages, the optimizer
(both Newton’s method and VBD) provides very small step sizes, so
it usually takes several iterations for the accumulated displacements
to exceed the conservative bounds. As a result, OGC-based methods
require significantly fewer collision detections, leading to much
lower computational time compared to IPC.

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

Fig. 14. Simulating a robot manipulating a T-shirt. The robot’s tra-
jectory is pre-computed. The T-shirt mesh has 13.8K vertices and 27.4K
triangles. We use a collision radius of 2mm and a time step of 1/600s
for the simulation. The average/maximum computation time per time
step is 1.8/2.2ms on the GPU.

This efficiency is particularly advantageous for VBD-based OGC.
Since VBD tends to generate smaller steps than Newton’s method, it
is less likely to exceed the conservative bounds, allowing the simula-
tion to fully leverage VBD’s output. Newton’s method, on the other
hand, tends to provide larger optimization steps, which supposedly
can lead to a more significant reduction in error. However, much
of this potential gain can be lost due to conservative bound culling,
resulting in wasted computation. Overall, on the CPU, VBD-based
OGC is more than about 128 times faster than IPC, while Newton’s
method-based OGC is 9.2 times faster than IPC on average.

VBD-based OGC is more advantageous over IPC on GPU. We
compare the GPU implementation of VBD-based OGC with GIPC
[Huang et al. 2024a], the state-of-the-art GPU variant of IPC. We
used the open-sourced implementation of GIPC to generate results.
For testing, we simulate the twisting of a volumetric mat at an
𝜋
angular velocity
2 by 16 seconds and evaluate the computational
time at each step. The results are visualized in Figure 19.

The average step time for GIPC is 1893ms, while VBD-OGC re-
quires only 5.51ms per step on average, making it 343 times faster
and capable of achieving near real-time performance, even under in-
tensive collisions and deformations. Furthermore, VBD-OGC demon-
strates significantly more stable performance, with the maximum
step time reaching only 8.5 ms. In contrast, GIPC’s maximum step
time exceeds 20 seconds, occurring at the end of the simulation when
the object experiences extensive self-contact, leading to minimal
optimization progress in each iteration. This comparison highlights
OGC’s suitability for real-time simulations due to its consistent and
efficient time consumption.

Offset Geometric Contact

•

17

n
o
t
w
e
N

s
r
u
O

Fig. 15. The yarn cloth is slowly pulled apart. Pure Newton is unable to prevent penetrations which cause catastrophic unraveling. In contrast, our
method is able to maintain a penetration free state through-out.

n
o
t
w
e
N

s
r
u
O

Fig. 16. A square yarn cloth is clamped on its edges and twisted 5 full rotations. Pure Newton is unable to keep the yarn threads separate as they
tangle. Our method is able to preserve the yarn structure despite the extreme deformation.

Fig. 17. Convergence plot of Newton’s method and VBD-based OGC for simulating three clothes dropping on a sphere at the given step, with 14.7K
vertices, 28.6K triangles, and a time step of 1/100𝑠. The graphs show relative force residuals change over iterations and computation time.

We also present the results of the same experiment using the
CPU implementations of VBD-OGC and IPC in the bottom row of
Figure 19. For IPC, we use its officially released implementation.

While IPC takes an average of 61.18 seconds per time step, the CPU
version of VBD-OGC completes each step in just 0.540 seconds

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

0100200300400500Iteration104103102101100Relative Force ResidualsVBDNewton0.00.10.20.30.40.50.6Time (s)104103102101100Relative Force ResidualsVBD-CPUVBD-GPUNewton18

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

Fig. 19. We compare the GPU implementation of VBD-OGC with
GIPC [Huang et al. 2024a], and the CPU implementation of VBD-OGC
with IPC, by replicating the volumetric mat twisting experiment pre-
sented in the IPC [Li et al. 2020] and GIPC [Huang et al. 2024a] papers.
The mesh has 15.3K vertices and 46.8K tets. The simulation is run with
a time step of 1/240s. The four images in the first row show the state of
the simulation using IPC at 0s, 4s, 8s, and 12s, respectively. The middle
row compares the runtime of each step in the GPU implementation
of VBD-OGC against GIPC, while the bottom row compares the CPU
implementation of VBD-OGC with IPC. We use a contact radius of 2
mm for OGC and allow GIPC and IPC to automatically control the
contact radius. We plot the time consumption at each step in the chart.

in Figure 2a and Figure 2b, respectively. The cloth consists of a
200 × 200 regular grid with each side measuring 1 meter, resulting in
a 0.5mm minimal distance between neighboring vertices. As shown
in Figure 2a, the IPC model produces severe artifacts caused by non-
orthogonal forces from neighbors and other points, including vertex
bulging and oscillations. In contrast, the OGC model handles the
large contact radius robustly, producing stable and natural contact
results. Please see the supplementary video for a more thorough
side-by-side comparison.

5.5.2 Numerical Damping. IPC is known to exhibit severe numeri-
cal damping artifacts when convergence is insufficient. This issue is
demonstrated in Figure 20, where a square cloth is simulated drop-
ping onto a small sphere, causing self-contact. In this experiment,
we use a time step of dt=1/500 but limit the solver to only one itera-
tion per step. Once self-contact occurs, IPC quickly loses nearly all

Fig. 18. We compare Newton’s method-based OGC and VBD-based
OGC with IPC by simulating a cloth dropping onto a fixed sphere. The
mesh has 4.9K vertices and 9.5K triangles. The simulation is run with
a time step of 1/100s for 100 steps. The four images in the first row
show the state of the simulation using IPC at steps 0, 25, 50, and 75,
respectively. We use a contact radius of 5mm for OGC and allow IPC to
automatically control the contact radius. From top to bottom, the first
figure illustrates the computational time at each step for each method,
the second one is the same as the first one but in a logarithmic scale,
and the third chart shows the number of collision detections used at
each step.

on average, achieving a 133× speedup. This demonstrates that our
method’s advantages do not only come from better parallelism.

5.5 Qualitative Comparison to Incremental Potential

Contact

5.5.1 Work with Large Contact Radius. We compare the compatibil-
ity of the IPC and OGC contact models with a large contact radius by
simulating a cloth twisted by half a circle using both methods. The
final states of the simulations using the IPC and OGC are shown

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

C
G
O

C
P
I

Fig. 20. Comparing our method (OGC+VBD) with IPC in the low
iteration count setup. In both of those experiments we use a time step
of 1/500s and only 1 iteration per step. The cloth has 4.9K vertices and
9.5K triangles. The top row is the results of OGC+VBD and the bottom
row is the results of IPC. From left to right, each column visualizes the
simulation state at frame 0, 50 and 80.

momentum, resulting in a slow-motion effect. This happens because
the self-contact restricts the optimization step size, allowing only
minimal movement per step and effectively dissipating velocity.

In contrast, the OGC model, with its conservative initialization
scheme and per-vertex-based displacement bounds, preserves the
momentum for most vertices, producing simulations with signifi-
cantly more dynamics. While neither IPC nor OGC achieves full nu-
merical convergence under such limited iterations, the OGC model
generates results that are far more visually plausible.

5.5.3 Comparing Activation Functions. To demonstrate the effec-
tiveness of our activation function, we conduct an ablation test,
with results visualized in Figure 21. In this experiment, we run two
simulations with our and IPC’s activation function, simulating a
piece of cloth dropped onto a sphere on the ground, with a time
step of 1/100s, collision stiffness 𝑘𝑐 = 1𝑒4, and VBD solver. We first
simulate 40 time steps, ensuring each step reaches numerical con-
vergence. The state of the simulation at the 40th step is visualized in
Figure 21a and Figure 21b. We can see that the simulations with two
activation functions provide visually identical results. Furthermore,
we plot the change in relative force residuals (Equation 29) over
iterations of the 40th step in Figure 21c, where the simulation using
our activation converges approximately 2x faster than the one using
IPC’s activation. At last, we plot the force-distance relationship of
two functions in Figure 21d, where we set the contact radius and col-
lision stiffness 𝑘𝑐 of both of those activations to be 1. Our activation
shows a smoother transition from 0 to infinity and exhibits less stiff
behavior. In fact, at the state visualized in Figure 21a and Figure 21b,
the condition number of the system Hessian of the simulation with
our activation is 5 times smaller than that using IPC’s activation.

6 LIMITATIONS AND FUTUREWORKS

Offset Geometric Contact is a contact model intended to achieve
orthogonality of normal contact force. However, on a discrete sur-
face, orthogonality and continuity of contact force cannot both be
achieved. As illustrated in Figure 22a, when a point moves along

Offset Geometric Contact

•

19

(a) OGC (ours)

(b) IPC

(c) Convergence Plot by Iterations

(d) Force-Distance Relationship for OGC (ours) and IPC

Fig. 21. Comparing our activation function Equation 18 with IPC’s
activation function. The results are measured on the 40th time step of
simulating a piece of cloth falling onto a sphere. The cloth has 4.9K
vertices and 9.5K triangles, and we use a time step of 1/100s. The state
of simulation at the selected step using OGC and IPC’s activation
function is visualized in (a) and (b), respectively. Panel (c) plots the
convergence of relative force residuals by iteration, and panel (d) shows
the force-distance relationship of the two functions.

the black trajectory, it is subject to discontinuous contact forces,
particularly upon entering the facet’s block from the open bound-
ary. At that moment, it suddenly experiences a non-zero contact
force from the facet. Note that this discontinuity only occurs at the
open boundary on the concave side of the faces, not at the closed
boundary, where the contact force is zero. The more concave the
area is, the more likely this issue is to arise, because it occurs in
the overlapping area of two adjacent face’s blocks. However, the
more concave the area is, the less likely it is for a point to enter the

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

050100150200250300Iteration104103102101100Relative Force ResidualsOGC (ours)IPC0.00.20.40.60.81.0Distance020406080100ForceOGC (ours)IPC20

• Anka He Chen, Jerry Hsu, Ziheng Liu, Miles Macklin, Yin Yang, and Cem Yuksel

(a) Discontinuity

(b) High velocity

Fig. 22. Illustration of our method’s limitations. (a) A point’s trajec-
tory that results in a discontinuous contact force. The thickened line
and dots represent the face, and the colored lines indicate the bound-
aries of the face’s block with corresponding color. Solid lines denote
closed boundaries, while dashed lines denote open boundaries. The
solid black line visualizes the trajectory. (b) Two mass points dropping
with high velocity from current position x𝑡 to the next time step’s
position x𝑡 +1, the circle visualizes the conservative bound given by
our method and the black arrow visualizes their trajectory within this
time step.

narrow space between faces, which helps mitigate the problem. In
theory, this discontinuity can lead to instability or slow convergence,
though we did not observe any such issues in our experiments.

Our technique for achieving penetration-free simulation is signifi-
cantly more efficient in scenarios with intensive collisions. However,
in cases with few collisions and large velocities, it may lag behind
the CCD-aware line search employed by IPC. As visualized in Fig-
ure 22b two mass points falling freely under gravity in parallel:
IPC’s technique can apply the full step in a single iteration because
their trajectories do not intersect. In contrast, our approach lim-
𝑑
2 , where 𝑑 is the distance between
its their motion to less than
the two points. This restriction can necessitate more iterations for
convergence, and the issue worsens as velocity increases.

Nevertheless, this limitation also suggests a promising direction
for future improvement. Potential strategies include intelligently
switching among various penetration-free techniques or incorpo-
rating vertex displacement directions to establish tighter bounds.

7 CONCLUSION

We have presented offset geometric contact, an efficient contact
model that allows for penetration-free simulation of codimensional
objects, significantly reducing the stiffness of contact forces and in-
creasing the efficiency of the simulation pipeline. By offsetting each
face in its normal directions, our formulation ensures normal con-
tact forces remain orthogonal, allowing for a larger contact distance
and thus mitigating the stiffness problem. Instead of continuous col-
lision detection (CCD), we compute a local maximum displacement
bound for each vertex in parallel, adding negligible overhead. This
local approach, combined with a fully parallel solver like Vertex
Block Descent, enables real-time, large-scale simulations on GPUs.
Our experiments show that this method can be more than two or-
ders of magnitude faster than IPC-based simulations and maintain
near-constant computational cost by using a fixed iteration count,

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

making penetration-free simulation feasible for a broader range of
applications.

Our results demonstrate that the proposed method effectively
handles highly complex simulation scenarios (Figure 10), maintains
stability under extreme stress tests (Figure 11, 12, 15, and 16), and
exhibits fast convergence (Figure 17).

In addition, we present an efficient implementation of our contact
model integrated with the VBD integrator, leveraging block-level
operations to maximize parallelism and efficiency. We aim to pro-
vide an out-of-the-box simulator and will continue to maintain the
code base after release. We also welcome collaboration with the
robotics, animation, and medical simulation communities to refine
the usability of penetration-free simulation, making it more accessi-
ble and beneficial for these fields. We hope these communities will
build upon our work to advance their respective applications.

ACKNOWLEDGMENTS

This project was supported in part by NSF grants #1956085 and
#2301040 and a gift from Meta.

REFERENCES

Oswin Aichholzer, Franz Aurenhammer, David Alberts, and Bernd Gärtner. 1996. A

novel type of skeleton for polygons. Springer.

Ryoichi Ando. 2024. A Cubic Barrier with Elasticity-Inclusive Dynamic Stiffness. ACM

Trans. Graph. 43, 6 (2024), 224:1–224:13. https://doi.org/10.1145/3687908

Thomas Banchoff. 1967. Critical points and curvature for embedded polyhedra. Journal

of Differential Geometry 1, 3-4 (1967), 245–256.

Thomas F Banchoff. 1970. Critical points and curvature for embedded polyhedral

surfaces. The American Mathematical Monthly 77, 5 (1970), 475–485.

Gill Barequet and Alex Goryachev. 2014. Offset polygon and annulus placement
problems. Computational Geometry 47, 3, Part A (2014), 407–434. https://doi.org/
10.1016/j.comgeo.2013.10.003

Qian Bo. 2010. Recursive polygon offset computing for rapid prototyping applications
based on Voronoi diagrams. The International Journal of Advanced Manufacturing
Technology 49 (2010), 1019–1028.

Ulrich Brehm and Wolfgang Kühnel. 1982. Smooth approximation of polyhedral surfaces

regarding curvatures. Geometriae Dedicata 12, 4 (1982), 435–461.

James V Burke. 1992. A robust trust region method for constrained nonlinear program-

ming problems. SIAM J. on Optim. 2, 2 (1992), 325–347.

James V Burke, Jorge J Moré, and Gerardo Toraldo. 1990. Convergence properties of
trust region methods for linear and convex constraints. Math. Program. 47, 1 (1990),
305–336.

Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024b. Vertex Block Descent.
ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages. https://doi.org/10.1145/
3658179

He Chen, Elie Diaz, and Cem Yuksel. 2023. Shortest Path to Boundary for Self-
Intersecting Meshes. ACM Trans. Graph. 42, 4, Article 146 (July 2023), 15 pages.
https://doi.org/10.1145/3592136

Honglin Chen, Hsueh-Ti Derek Liu, Alec Jacobson, David I. W. Levin, and Changxi
Zheng. 2024a. Trust-Region Eigenvalue Filtering for Projected Newton. In SIG-
GRAPH Asia 2024 Conference Papers, SA 2024, Tokyo, Japan, December 3-6, 2024,
Takeo Igarashi, Ariel Shamir, and Hao (Richard) Zhang (Eds.). ACM, 120:1–120:10.
https://doi.org/10.1145/3680528.3687650

Xiaorui Chen and Sara McMains. 2005. Polygon offsetting by computing winding
numbers. In International Design Engineering Technical Conferences and Computers
and Information in Engineering Conference, Vol. 4739. 565–575.

Jonathan Cohen, Marc Olano, and Dinesh Manocha. 1998. Appearance-preserving
simplification. In Proceedings of the 25th annual conference on Computer graphics
and interactive techniques. 115–122.

Andrew R Conn, Nicholas IM Gould, and Ph L Toint. 1988. Global convergence of a
class of trust region algorithms for optimization with simple bounds. SIAM journal
on numerical analysis 25, 2 (1988), 433–460.

Gilberto Echeverria. 2007. The Polyhedral Gauss Map and discrete curvature measures in
geometric modelling. Ph. D. Dissertation. Sheffield Hallam University, UK. https:
//ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.440302

Zachary Ferguson, Minchen Li, Teseo Schneider, Francisca Gil-Ureta, Timothy Langlois,
Chenfanfu Jiang, Denis Zorin, Danny M. Kaufman, and Daniele Panozzo. 2021.
Intersection-free rigid body dynamics. ACM Trans. Graph. 40, 4, Article 183 (July

𝒙𝒙t𝒙𝒙t+1Offset Geometric Contact

•

21

2021), 16 pages. https://doi.org/10.1145/3450626.3459802

//doi.org/10.1145/3450626.3459767

Dewen Guo, Minchen Li, Yin Yang, Sheng Li, and Guoping Wang. 2024. Barrier-
Augmented Lagrangian for GPU-based Elastodynamic Contact. ACM Trans. Graph.
43, 6 (2024), 225:1–225:17. https://doi.org/10.1145/3687988

Berthold Klaus Paul Horn. 1984. Extended gaussian images. Proc. IEEE 72, 12 (1984),

1671–1686.

Kemeng Huang, Floyd Chitalu, Huancheng Lin, and Taku Komura. 2024a. GIPC: Fast
and stable Gauss-Newton optimization of IPC barrier energy. http://arxiv.org/abs/
2308.09400 arXiv:2308.09400 [cs] version: 4.

Kemeng Huang, Floyd M. Chitalu, Huancheng Lin, and Taku Komura. 2024b. GIPC:
Fast and Stable Gauss-Newton Optimization of IPC Barrier Energy. ACM Trans.
Graph. 43, 2, Article 23 (Mar 2024), 18 pages. https://doi.org/10.1145/3643028
Stefan Huber. 2018. The topology of skeletons and offsets. In Proc. 34th Europ. Workshop

on Comp. Geom.(EuroCG’18).

T. Kugelstadt and E. Schömer. 2016. Position and orientation based Cosserat rods. In
Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation
(Zurich, Switzerland) (SCA ’16). Eurographics Association, Goslar, DEU, 169–178.
Lei Lan, Danny M. Kaufman, Minchen Li, Chenfanfu Jiang, and Yin Yang. 2022a. Affine
Body Dynamics: Fast, Stable and Intersection-Free Simulation of Stiff Materials.
ACM Trans. Graph. 41, 4, Article 67 (Jul 2022), 14 pages. https://doi.org/10.1145/
3528223.3530064

Lei Lan, Minchen Li, Chenfanfu Jiang, Huamin Wang, and Yin Yang. 2023. Second-order
Stencil Descent for Interior-point Hyperelasticity. ACM Trans. Graph. 42, 4, Article
108 (Jul 2023), 16 pages. https://doi.org/10.1145/3592104

Lei Lan, Zixuan Lu, Jingyi Long, Chun Yuan, Xuan Li, Xiaowei He, Huamin Wang,
Chenfanfu Jiang, and Yin Yang. 2024. Mil2: Efficient Cloth Simulation Using Non-
distance Barriers and Subspace Reuse. CoRR abs/2403.19272 (2024). https://doi.org/
10.48550/ARXIV.2403.19272 arXiv:2403.19272

Lei Lan, Guanqun Ma, Yin Yang, Changxi Zheng, Minchen Li, and Chenfanfu Jiang.
2022b. Penetration-free projective dynamics on the GPU. ACM Trans. Graph. 41, 4,
Article 69 (July 2022), 16 pages. https://doi.org/10.1145/3528223.3530069

Lei Lan, Yin Yang, Danny Kaufman, Junfeng Yao, Minchen Li, and Chenfanfu Jiang.
2021. Medial IPC: Accelerated incremental potential contact with medial elastics.
ACM Trans. Graph. 40, 4, Article 158 (Jul 2021), 16 pages. https://doi.org/10.1145/
3450626.3459753

Minchen Li, Danny M. Kaufman, and Chenfanfu Jiang. 2021. Codimensional incremental
potential contact. ACM Trans. Graph. 40, 4, Article 170 (July 2021), 24 pages. https:

Minchen Li et al. 2020. Incremental potential contact: intersection-and inversion-free,
large-deformation dynamics. ACM Trans. Graph. 39, 4, Article 49 (August 2020),
20 pages. https://doi.org/10.1145/3386569.3392425

James J Little. 1985. Extended gaussian images, mixed volumes, shape reconstruction.
In Proceedings of the first annual symposium on Computational geometry. 15–23.
Miles Macklin. 2022. Warp: A High-performance Python Framework for GPU Simu-
lation and Graphics. https://github.com/nvidia/warp. NVIDIA GPU Technology
Conference (GTC).

Jorge J Moré. 1983. Recent developments in algorithms and software for trust region

methods. Math. Program. The State of the Art: Bonn 1982 (1983), 258–287.

Jorge Nocedal and Stephen J Wright. 1999. Numerical optimization.
Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A.
Osman, Dimitrios Tzionas, and Michael J. Black. 2019. Expressive Body Capture:
3D Hands, Face, and Body From a Single Image. In IEEE Conf. on Comput. Vis. and
Pattern Recognit., CVPR 2019, Long Beach, CA, USA, June 16-20, 2019. 10975–10985.
https://doi.org/10.1109/CVPR.2019.01123

Xing Shen, Runyuan Cai, Mengxiao Bi, and Tangjie Lv. 2024. Preconditioned Nonlinear
Conjugate Gradient Method for Real-time Interior-point Hyperelasticity. In ACM
SIGGRAPH 2024 Conference Papers. 1–11.

Min Tang, Young J. Kim, and Dinesh Manocha. 2009. C2A: Controlled conservative
advancement for continuous collision detection of polygonal models. In 2009 IEEE
Int. Conf. on Robot. and Automat., ICRA 2009, Kobe, Jpn., May 12-17, 2009. IEEE,
849–854. https://doi.org/10.1109/ROBOT.2009.5152234

Tianyu Wang, Jiong Chen, Dongping Li, Xiaowei Liu, Huamin Wang, and Kun Zhou.
2023. Fast GPU-Based Two-Way Continuous Collision Handling. ACM Trans. Graph.
(SIGGRAPH) 42, 5, Article 167 (Jul 2023), 15 pages. https://doi.org/10.1145/3604551
Botao Wu, Zhendong Wang, and Huamin Wang. 2022. A GPU-based multilevel additive
schwarz preconditioner for cloth and deformable body simulation. ACM Trans.
Graph. 41, 4 (2022), 63:1–63:14. https://doi.org/10.1145/3528223.3530085

Longhua Wu, Botao Wu, Yin Yang, and Huamin Wang. 2020. A Safe and Fast Repulsion
Method for GPU-based Cloth Self Collisions. ACM Trans. Graph. (SIGGRAPH) 40, 1,
Article 5 (Dec 2020), 18 pages. https://doi.org/10.1145/3430025

Ya-Xiang Yuan. 2015. Recent advances in trust region algorithms. Math. Program. 151,

1 (2015), 249–281. https://doi.org/10.1007/S10107-015-0893-2

Xinyu Zhang, Minkyoung Lee, and Young J Kim. 2006. Interactive continuous collision

detection for non-convex polyhedra. The Vis. Comput. 22 (2006), 749–760.

ACM Trans. Graph., Vol. 44, No. 4, Article . Publication date: August 2025.

