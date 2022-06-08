# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from pyparsing import java_style_comment
import scipy
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    if cell is ReferenceInterval:
        x = np.zeros((degree+1,1))
        for i in range(0,degree+1):
            x[i]=i/degree
        return x
    elif cell is ReferenceTriangle:
        num_points = int(scipy.special.binom(degree+2,2))
        x = np.zeros((num_points,2))
        k=0
        for i in range(0,degree+1):
            for j in range(0,degree+1-i):
                x[k] = [j/degree,i/degree]
                k += 1
    else:
        raise NotImplementedError

    return x



def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """
    points = np.array(points)
    if grad is True:
        if cell is ReferenceInterval:
            #v = np.ndarray(shape=(degree+2,degree+1), dtype=float)
            V = np.zeros((points.shape[0],degree+1,1))
            V[:,0] = 0
            for j in range(1,degree+1):
                for i in range(0,points.shape[0]):
                    V [i,j] = j*points[i]**(j-1)
        elif cell is ReferenceTriangle:
            num_points = points.shape[0]
            num_terms = int((degree+1)*(degree+2)/2)
            V = np.zeros((num_points,num_terms,2))
            for k in range(0,num_points):
                m=0
                for j in range (0,degree+1):
                    for i in range(0,j+1):
                        i = np.float64(i)
                        j = np.float64(j)
                        V[k,m] = np.nan_to_num([(j-i)*points[k,0]**(j-i-1)*points[k,1]**i,points[k,0]**(j-i)*i*points[k,1]**(i-1)])
                        m+=1
    if grad is False:
        if cell is ReferenceInterval:
            #v = np.ndarray(shape=(degree+2,degree+1), dtype=float)
            V = np.zeros((points.shape[0],degree+1))
            for j in range(0,degree+1):
                for i in range(0,points.shape[0]):
                    V [i,j] = points[i,0]**j
        elif cell is ReferenceTriangle:
            num_points = points.shape[0]
            num_terms = int((degree+1)*(degree+2)/2)
            V = np.zeros((num_points,num_terms))
            for k in range(0,num_points):
                m=0
                for j in range (0,degree+1):
                    for i in range(0,j+1):
                        points[k,0]
                        points[k,1]
                        V[k,m]
                        V[k,m] = points[k,0]**(j-i)*points[k,1]**i
                        m+=1
    return V    

class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        V= vandermonde_matrix(cell, degree, nodes)
        self.basis_coefs = np.linalg.inv(V)

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        V = vandermonde_matrix(self.cell, self.degree, points, grad)
        if grad is True:
            T = np.einsum('ijk,jl->ilk', V, self.basis_coefs)

        if grad is False:  
            T = np.dot(V,self.basis_coefs)
        return T



    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        return [fn(x) for x in self.nodes]

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell, degree)
        entity_nodes = {d: {e: [] for e in range(cell.entity_counts[d])} for d in range(cell.dim+1) }

        entities=[(d, e)
                 for d in cell.topology.keys()
                 for e in cell.topology[d].keys()]

        for i, n in enumerate(nodes):
            for (d, e) in entities: 
                if cell.point_in_entity(n, (d, e)):
                    entity_nodes[d][e].append(i)
                    break

        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)
