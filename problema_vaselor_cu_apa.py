import os
import sys
import copy
import time


class TraversalNode:
    def __init__(self, information: list, parent, cost: int = 0, h: int = 0):
        self.information = information
        self.parent = parent
        self.g = cost
        self.h = h
        self.f = self.g + self.h

    def get_path(self) -> list:  # drumul pana la nodul curent
        path = [self]
        node = self
        while node.parent is not None:
            path.insert(0, node.parent)
            node = node.parent
        return path

    def print_path(self) -> int:  # afiseaza drumul pana la nodul curent
        global output_file
        path = self.get_path()
        for i in range(len(path)):
            output_file.write("Nodul {}\n".format(i + 1))
            output_file.write(str(path[i]) + "\n")

        output_file.write("Lungime: {}\n".format(len(path)))
        output_file.write("Cost: {}\n".format(self.f))
        return len(path)

    def contains_in_path(self, node_info: list) -> bool:  # verfica daca node_info este informatia unui nod continut
        node = self  # deja in drum
        while node is not None:
            if node_info == node.information:
                return True
            node = node.parent
        return False

    def __str__(self):
        output_string = ""
        if self.parent is not None:
            parent_vessels = copy.deepcopy(self.parent.information)
            vessels = copy.deepcopy(self.information)
            give_vessel = take_vessel = 0
            litres = 0
            color = ""
            for i in range(len(vessels)):
                if vessels[i][1] < parent_vessels[i][1]:  # vasul din care torn apa + culoarea apei
                    give_vessel = i
                    if vessels[i][2] == "":
                        color = parent_vessels[i][2]
                    else:
                        color = vessels[i][2]
                elif vessels[i][1] > parent_vessels[i][1]:  # vasul care primeste apa + cantitatea primita
                    take_vessel = i
                    litres = vessels[i][1] - parent_vessels[i][1]

            output_string = "Din vasul {} s-au turnat {} litri de apa de culoare {} in vasul {}\n". \
                format(give_vessel, litres, color, take_vessel)

        for i in range(len(self.information)):  # afisez configuratia vaselor
            vessel = self.information[i]
            if len(vessel) == 3:
                output_string += "{}: {} {} {}\n".format(i, *vessel)
            else:
                output_string += "{}: {} {}\n".format(i, *vessel)
        return output_string


class Graph:
    def __init__(self, input_file: str):
        f = open(input_folder + "/" + input_file, 'r')
        content = f.read().split("stare_initiala")
        f.close()
        initial = content[0].strip().split('\n')
        self.combinations = []
        self.costs = {}
        colors = set()  # culorile folosite in combinatii
        for s in initial:
            t = tuple(s.strip().split())
            if len(t) == 3:  # tupluri care contin combinatii de culori
                if t[0] > t[1]:  # ordonez culorile combinate alfabetic
                    t = (t[1], t[0], t[2])
                    # validare
                    colors.add(t[0])  # adaug culorile in set
                    colors.add(t[1])
                    colors.add(t[2])
                self.combinations.append(t)
            else:
                self.costs[t[0]] = int(t[1])  # costurile asociate culorilor
                # validare
                if int(t[1]) < 0:
                    print("Costul unei culori nu poate fi negativ!")
                    exit()
        # validare
        for c in colors:
            if self.costs.get(c) is None:  # pentru culorile care apar, dar nu s-a specificat costul, acesta este 1
                self.costs[c] = 1

        self.combinations.sort(key=lambda k: (k[0], k[1]))
        self.costs["nedefinit"] = 1  # adaug costuri pentru culoarea nedefinita si culoarea vida
        self.costs[""] = 0

        initial_stage, final_stage = content[1].split("stare_finala")
        initial_stage = initial_stage.strip().split('\n')
        final_stage = final_stage.strip().split('\n')
        self.start = [tuple(v.split()) for v in initial_stage]
        scopes = [tuple(v.split()) for v in final_stage]

        for i in range(len(self.start)):  # starea initiala a grafului contine
            if len(self.start[i]) == 2:  # tupluri de forma (capacitate, cantitate, culoare)
                capacity, litres = self.start[i]
                self.start[i] = (int(capacity), int(litres), "")  # pentru un vas gol, culoarea este ""
            else:
                capacity, litres, color = self.start[i]
                self.start[i] = (int(capacity), int(litres), color)
            # validare
            if self.start[i][0] <= 0 or self.start[i][1] < 0:
                print("Capacitatile vaselor si cantitatile de apa nu pot fi numere negative!")
                exit()

        self.scopes = {}
        for i in range(len(scopes)):  # starea scop este un dictionar in care se retin cantitatile dorite
            capacity, color = scopes[i]  # din anumite culori
            self.scopes[color] = int(capacity)
            # validare
            if int(capacity) < 0:
                print("Cantitatile dorite la final nu pot fi negative!")
                exit()

        # global output_file
        # output_file.write("Combinari:\n")
        # for v in self.combinations:
        #     output_file.write("\t" + str(v) + "\n")
        # output_file.write("Costuri:\n")
        # for c in self.costs.keys():
        #     output_file.write("\t" + c + " -> " + str(self.costs[c]) + "\n")
        # output_file.write("Starea initiala:\n")
        # for x in self.start:
        #     output_file.write("\t" + str(x) + "\n")
        # output_file.write("Starea finala:\n")
        # for c in self.scopes.keys():
        #     output_file.write("\t" + c + " -> " + str(self.scopes[c]) + "\n")

    def test_scope(self, current_node: TraversalNode) -> bool:
        stage = current_node.information
        colors = {}
        for vessel in stage:  # calculez cantitatile din fiecare culoare pentru nodul curent
            if vessel[2] in colors.keys():
                colors[vessel[2]] += vessel[1]
            else:
                colors[vessel[2]] = vessel[1]

        for k in self.scopes.keys():  # verific corespondenta dintre cantitatile de culoare
            if (k not in colors.keys()) or self.scopes[k] != colors[k]:
                return False
        return True

    def generate_successors(self, current_node: TraversalNode, heuristic_type: str = "euristica banala") -> list:
        successors_list = []
        vessels = current_node.information
        number_of_vessels = len(vessels)
        for i in range(number_of_vessels):
            vessels_copy = copy.deepcopy(vessels)
            if vessels_copy[i][1] == 0:  # daca vasul i este gol, trec la urmatorul vas
                continue
            # mut lichid din vasul i in vasul j
            for j in range(number_of_vessels):
                if i == j:
                    continue
                new_vessels = copy.deepcopy(vessels_copy)
                litres = min(new_vessels[i][1], new_vessels[j][0] - new_vessels[j][1])
                if litres == 0:
                    continue
                new_vessels[i] = (new_vessels[i][0], new_vessels[i][1] - litres, new_vessels[i][2])
                new_vessels[j] = (new_vessels[j][0], new_vessels[j][1] + litres, new_vessels[j][2])
                new_cost = litres * self.costs[new_vessels[i][2]]
                if new_vessels[j][2] == "":  # calculez costul
                    new_cost = litres * self.costs[new_vessels[i][2]]
                    new_vessels[j] = (new_vessels[j][0], new_vessels[j][1], new_vessels[i][2])
                elif new_vessels[i][2] == "nedefinit" or new_vessels[j][2] == "nedefinit":
                    new_cost = litres * self.costs[new_vessels[i][2]] + \
                               (new_vessels[j][1] - litres) * self.costs[new_vessels[j][2]]
                    new_vessels[j] = (*new_vessels[j][:2], "nedefinit")
                else:
                    combination = ()
                    if new_vessels[i][2] == new_vessels[j][2]:
                        pass
                    elif new_vessels[i][2] < new_vessels[j][2]:
                        combination = (new_vessels[i][2], new_vessels[j][2])
                    else:
                        combination = (new_vessels[j][2], new_vessels[i][2])

                    if combination != ():
                        left = 0
                        right = len(self.combinations) - 1
                        index = None
                        while left <= right:
                            middle = (left + right) // 2
                            if combination == self.combinations[middle][:2]:
                                index = middle
                                break
                            elif combination < self.combinations[middle][:2]:
                                right = middle - 1
                            else:
                                left = middle + 1

                        if index is not None:
                            new_cost = litres * self.costs[new_vessels[i][2]]
                            new_vessels[j] = (new_vessels[j][0], new_vessels[j][1], self.combinations[index][2])
                        else:
                            new_cost = litres * self.costs[new_vessels[i][2]] + \
                                       (new_vessels[j][1] - litres) * self.costs[new_vessels[j][2]]
                            new_vessels[j] = (new_vessels[j][0], new_vessels[j][1], "nedefinit")

                if new_vessels[i][1] == 0:
                    new_vessels[i] = (new_vessels[i][0], new_vessels[i][1], "")

                if not current_node.contains_in_path(new_vessels):
                    successors_list.append(TraversalNode(new_vessels, current_node, current_node.g + new_cost))

        return successors_list


def obtain_color(color: str, initial_colors: list, colors_combinations: list) -> bool:
    if color in initial_colors:
        return True
    colors_combinations.sort(key=lambda t: t[2])
    left = 0
    right = len(colors_combinations) - 1
    index = None
    while left <= right:
        middle = (left + right) // 2
        if colors_combinations[middle][2] == color:
            index = middle
            break
        elif colors_combinations[middle][2] < color:
            left = middle + 1
        else:
            right = middle - 1
    if index is None:
        return False
    return obtain_color(colors_combinations[index][0], initial_colors, colors_combinations) and \
        obtain_color(colors_combinations[index][1], initial_colors, colors_combinations)


# TODO:functie pentru a verifica daca graful initial are scop(culori) + functie care determina asta pentru TraversalNode
def has_scope(final_colors: list, initial_colors: list, colors_combinations: list) -> bool:
    if not final_colors:
        return True
    obtained_colors = copy.deepcopy(initial_colors)
    for c in final_colors:
        if c in obtained_colors:
            final_copy = copy.deepcopy(final_colors[1:])
            return has_scope(final_copy, obtained_colors, colors_combinations)
        else:
            if obtain_color(c, obtained_colors, colors_combinations):
                obtained_colors.append(c)
                final_copy = copy.deepcopy(final_colors[1:])
                return has_scope(final_copy, obtained_colors, colors_combinations)
            else:
                return False
    return False


def breadth_first(source_graph: Graph, searched_solutions: int) -> None:
    global output_file, number_of_solutions
    start_time = time.time()
    queue = [TraversalNode(source_graph.start, None)]  # coada cu noduri
    max_nodes = 1
    total_nodes = 0
    while len(queue) > 0:
        max_nodes = max(max_nodes, len(queue))
        current_node = queue.pop(0)  # extrag primul nod din coada
        if source_graph.test_scope(current_node):  # daca nodul curent este nod scop, afisez solutia
            intermediate_time = time.time()
            output_file.write("----BREADTH FIRST----\n\nSolutia {}:\n".
                              format(number_of_solutions + 1 - searched_solutions))
            current_node.print_path()
            output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
            output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                              format(max_nodes, total_nodes))
            max_nodes = 1
            total_nodes = 0
            output_file.write("\n---------------------------------------------------------------------------------\n\n")
            searched_solutions -= 1
            if searched_solutions == 0:
                return

        successors = source_graph.generate_successors(current_node)
        total_nodes += len(successors)
        queue.extend(successors)  # adaug in coada succesorii nodului curent


def uniform_cost_search(source_graph: Graph, searched_solutions: int) -> None:
    global output_file, timeout, number_of_solutions
    start_time = time.time()
    queue = [TraversalNode(source_graph.start, None)]  # coada cu noduri
    used = [source_graph.start]
    max_nodes = 1
    total_nodes = 0
    while len(queue) > 0:
        max_nodes = max(max_nodes, len(queue))
        current_node = queue.pop(0)  # extrag primul nod din coada
        if source_graph.test_scope(current_node):  # daca nodul curent este nod scop, afisez solutia
            intermediate_time = time.time()
            output_file.write("----UNIFORM COST SEARCH----\n\nSolutia {}:\n".
                              format(number_of_solutions + 1 - searched_solutions))
            current_node.print_path()
            output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
            output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                              format(max_nodes, total_nodes))
            max_nodes = 1
            total_nodes = 0
            output_file.write("\n---------------------------------------------------------------------------------\n\n")
            searched_solutions -= 1
            if searched_solutions == 0:
                return

        intermediate_time = time.time()
        if intermediate_time - start_time > timeout:
            output_file.write("Timpul de executie a depasit timeout-ul!\n")
            break

        successors = source_graph.generate_successors(current_node)
        total_nodes += len(successors)
        for s in successors:  # adaug succesorii in coada astfel incat aceasta sa ramana sortata crescator dupa cost
            if s.information in used:
                continue
            pos = 0
            found_place = False
            for pos in range(len(queue)):
                if queue[pos].g > s.g:
                    found_place = True
                    break
            if found_place:
                queue.insert(pos, s)
            else:
                queue.append(s)
            used.append(s.information)


def a_star(source_graph: Graph, searched_solutions: int) -> None:
    global output_file, timeout, number_of_solutions
    start_time = time.time()
    queue = [TraversalNode(source_graph.start, None)]  # coada cu noduri
    used = [source_graph.start]
    max_nodes = 1
    total_nodes = 0
    while len(queue) > 0:
        max_nodes = max(max_nodes, len(queue))
        current_node = queue.pop(0)  # extrag primul nod din coada
        if source_graph.test_scope(current_node):  # daca nodul curent este nod scop, afisez solutia
            intermediate_time = time.time()
            output_file.write("----A*----\n\nSolutia {}:\n".format(number_of_solutions + 1 - searched_solutions))
            current_node.print_path()
            output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
            output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                              format(max_nodes, total_nodes))
            max_nodes = 1
            total_nodes = 0
            output_file.write("\n---------------------------------------------------------------------------------\n\n")
            searched_solutions -= 1
            if searched_solutions == 0:
                return

        intermediate_time = time.time()
        if intermediate_time - start_time > timeout:
            output_file.write("Timpul de executie a depasit timeout-ul!\n")
            break

        successors = source_graph.generate_successors(current_node)
        total_nodes += len(successors)
        for s in successors:  # adaug succesorii in coada astfel incat aceasta sa ramana sortata crescator dupa cost
            if s.information in used:
                continue
            pos = 0
            found_place = False
            for pos in range(len(queue)):
                if queue[pos].f >= s.f:
                    found_place = True
                    break
            if found_place:
                queue.insert(pos, s)
            else:
                queue.append(s)
            used.append(s.information)


def a_star_optimised(source_graph: Graph) -> None:
    global output_file, timeout
    start_time = time.time()
    open_list = [TraversalNode(source_graph.start, None)]
    closed_list = []
    max_nodes = 1
    total_nodes = 0
    while len(open_list) > 0:
        max_nodes = max(max_nodes, len(open_list) + len(closed_list))
        current_node = open_list.pop(0)
        closed_list.append(current_node)
        if source_graph.test_scope(current_node):
            intermediate_time = time.time()
            output_file.write("----A* OPTIMISED----\n\nSolutie:\n")
            current_node.print_path()
            output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
            output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                              format(max_nodes, total_nodes))
            output_file.write("\n---------------------------------------------------------------------------------\n\n")
            return

        intermediate_time = time.time()
        if intermediate_time - start_time > timeout:
            output_file.write("Timpul de executie a depasit timeout-ul!\n")
            break

        successors = source_graph.generate_successors(current_node)
        total_nodes += len(successors)
        for s in successors[:]:
            found_in_open = False
            for node in open_list:
                if s.information == node.information:
                    found_in_open = True
                    if s.f >= node.f:
                        successors.remove(s)
                    else:
                        open_list.remove(node)
            if not found_in_open:
                for node in closed_list:
                    if s.information == node.information:
                        if s.f >= node.f:
                            successors.remove(s)
                        else:
                            closed_list.remove(node)

        for s in successors:
            pos = 0
            found_place = False
            for pos in range(len(open_list)):
                if open_list[pos].f > s.f or (open_list[pos].f == s.f and open_list[pos].g <= s.g):
                    found_place = True
                    break
            if found_place:
                open_list.insert(pos, s)
            else:
                open_list.append(s)


def iterative_deepening_a_star(source_graph: Graph, searched_solutions: int) -> None:
    global output_file, timeout
    start_time = time.time()
    start_node = TraversalNode(source_graph.start, None)
    limit = start_node.f
    while True:
        searched_solutions, result = build_path(source_graph, start_node, limit, searched_solutions, start_time)

        intermediate_time = time.time()
        if intermediate_time - start_time > timeout:
            output_file.write("Timpul de executie a depasit timeout-ul!\n")
            break

        if result == "done":
            break
        if result == float("inf"):
            output_file.write("Nu exista solutii!\n")
            break
        limit = result
        # print("------------------Noua limita " + str(limit))
        # input()


def build_path(source_graph: Graph, current_node: TraversalNode, limit: int, searched_solutions: int,
               start_time: float) -> tuple:
    global output_file, timeout, number_of_solutions
    max_nodes = 1
    total_nodes = 0
    if current_node.f > limit:
        return searched_solutions, current_node.f
    if source_graph.test_scope(current_node) and current_node.f == limit:
        intermediate_time = time.time()
        output_file.write("----ITERATIVE DEEPENING A*----\n\nSolutia {}:\n".
                          format(number_of_solutions + 1 - searched_solutions))
        current_node.print_path()
        output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
        output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                          format(max_nodes, total_nodes))
        output_file.write("\n---------------------------------------------------------------------------------\n\n")
        searched_solutions -= 1
        if searched_solutions == 0:
            return 0, "done"

    successors = source_graph.generate_successors(current_node)
    total_nodes += len(successors)
    minimum = float("inf")
    for s in successors:
        searched_solutions, result = build_path(source_graph, s, limit, number_of_solutions, start_time)
        if result == "done":
            return 0, "done"
        if result < minimum:
            minimum = result
            # print("Noul minim: " + str(minimum))

    return searched_solutions, minimum


def valid_arguments():
    global input_folder, number_of_solutions, timeout

    ok = True
    if not os.path.exists(input_folder):
        ok = False
        print("Folderul pentru input nu exista!")
    if number_of_solutions <= 0:
        ok = False
        print("Numarul de solutii cautate trebuie sa fie un numar natural diferit de 0!")
    if timeout <= 0:
        ok = False
        print("Timpul maxim de rulare nu poate fi negativ!")
    return ok


if __name__ == "__main__":
    _, input_folder, output_folder, number_of_solutions, timeout = sys.argv  # argumentele programului

    number_of_solutions = int(number_of_solutions)
    timeout = float(timeout)

    if not valid_arguments():
        exit()

    if not os.path.exists(output_folder):  # creez folderul pentru output daca nu exista
        os.mkdir(output_folder)

    input_files = os.listdir(input_folder)  # fisierele de input
    for file in input_files:
        output_file = open(output_folder + "/output_" + file, 'w')  # creez fisierul corespunzator de output
        graph = Graph(file)
        # breadth_first(graph, number_of_solutions)
        uniform_cost_search(graph, number_of_solutions)
        a_star(graph, number_of_solutions)
        a_star_optimised(graph)
        # iterative_deepening_a_star(graph, number_of_solutions)
        output_file.close()
        break
