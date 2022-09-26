## Shunted Collision Avoidance (SCA) for Multi-Agent Motion Planning in Three Dimensions

Python Implementation of shunted collision avoidance for multiple fixed-wing UAVs motion planning, including our method, namely shunted collision avoidance (SCA), the reciprocal volocity obstacles in 3D domains (RVO3D), and the  optimal reciprocal collision avoidance in 3D domains (ORCA3D) .

-----

Description
-----

We present an approach for fixed-wing UAVs' collision avoidance, where multiple independent mobile UAVs or agents need to avoid collisions without communication among agents while moving in a common 3D workspace. We named the proposed method as shunted collision avoidance (SCA). In addition, we provide the optimal reciprocal collision avoidance (ORCA) method and reciprocal velocity obstacles (RVO) method in 3D domains for comparison.

About
-----

**Paper**:  Shunted Collision Avoidance for Multi-UAV Motion Planning with Posture Constraints, Gang Xu, Deye Zhu, Junjie Cao\*,  Yong Liu\*, and Jian Yang


-----

Requirement
-----

```python
pip install numpy
pip install open3d
pip install pandas
pip install matplotlib
```

-----

Applications
-----

```python
cd run_example
python run_sca.py
For the test, you can select the scenarios, including circle, random, take-off and landing, etc.
```

#### The scenario of circle for simulating air patrol task.

<p align="center">
    <img src="visualization/figs/figexp11.png" width="800" height="400" />
</p>

<p align="center">
    <img src="visualization/figs/figexp12.png" width="800" height="400" />
</p>



#### The scenario of take-off and landing.

<p align="center">
    <img src="visualization/figs/figexp21.png" width="800" height="400" />
</p>

<p align="center">
    <img src="visualization/figs/figexp22.png" width="800" height="400" />
</p>



#### The scenario of circle for simulating low altitude flying for search task.

<p align="center">
    <img src="visualization/figs/figexp31.png" width="800" height="400" />
</p>

<p align="center">
    <img src="visualization/figs/figexp32.png" width="800" height="400" />
</p>



#### Results of Comparison 1 (Ours: S-RVO3D):

<p align="center">
    <img src="visualization/figs/c1.png" width="800" height="600" />
</p>


#### Results of Comparison 2 (Ours: SCA):

<p align="center">
    <img src="visualization/figs/c2.png" width="800" height="500" />
</p>





----

Discussion
----

In the first comparison, the UAVs' number is 100.
