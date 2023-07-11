from graphviz import Digraph

class PlotGraph:

    def trace(self, root):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    def draw_dot(self, root, format='svg', rankdir='LR'):
        assert rankdir in ['LR', 'TB']
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
        
        nodes, edges = self.trace(root)
        for n in nodes:
            dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')

            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        return dot