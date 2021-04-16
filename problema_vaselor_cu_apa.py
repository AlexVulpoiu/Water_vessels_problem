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
        output_file.write("Cost: {}\n".format(self.g))
        return len(path)

    def contains_in_path(self, node_info: list) -> bool:  # verfica daca node_info este informatia unui nod continut
        """
        :param node_info: list, configuratia nodului curent
        :return: bool, True daca exista deja un nod cu aceeasi informatie pe drumul pana la radacina
        """
        node = self
        while node is not None:
            if node_info == node.information:
                return True
            node = node.parent
        return False

    def __str__(self):
        output_string = ""
        if self.parent is not None:     # daca nodul curent nu este radacina
            parent_vessels = copy.deepcopy(self.parent.information)
            vessels = copy.deepcopy(self.information)
            give_vessel = take_vessel = -1  # calculez indicele vaselor din care se toarna, respectiv se primeste lichid
            litres = 0  # cantitatea mutata
            color = ""  # culoarea apei
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
        global output_file, valid_data

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
                    output_file.write("Costul unei culori nu poate fi negativ!\n")
                    valid_data = False
        # validare
        for c in colors:
            if self.costs.get(c) is None:  # pentru culorile care apar, dar nu s-a specificat costul, acesta este 1
                self.costs[c] = 1

        self.combinations.sort(key=lambda k: (k[0], k[1]))  # sortez combinatiile alfabetic dupa culorile amestecate
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
            if self.start[i][0] <= 0 or self.start[i][1] < 0 or self.start[i][1] > self.start[i][0]:
                output_file.write("Capacitatile vaselor si cantitatile de apa nu pot fi numere negative!\n")
                valid_data = False

        self.scopes = {}
        for i in range(len(scopes)):  # starea scop este un dictionar in care se retin cantitatile dorite
            capacity, color = scopes[i]  # din anumite culori
            self.scopes[color] = int(capacity)
            # validare
            if int(capacity) < 0:
                output_file.write("Cantitatile dorite la final nu pot fi negative!\n")
                valid_data = False

        aux_start = copy.deepcopy(self.start)
        aux_scopes = copy.deepcopy(self.scopes)
        aux_combinations = copy.deepcopy(self.combinations)
        if not has_scope(aux_start, aux_scopes, aux_combinations):   # verific la inceput daca nu exista solutii
            output_file.write("Nu exista solutii pentru datele primite!\n")
            valid_data = False

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
        """
        :param current_node: TraversalNode, nodul curent din arborele de parcurgere
        :return: bool, True daca este nod scop, False altfel
        """
        stage = current_node.information    # informatia nodului curent
        colors = {}     # colors[i] = cantitatea culorii i
        for vessel in stage:  # calculez cantitatile din fiecare culoare pentru nodul curent
            if vessel[2] in self.scopes.keys():     # calculez cantitatile din culorile care ma intereseaza
                if vessel[2] in colors.keys():
                    return False    # inseamna ca am mai gasit culoarea si in alt vas
                else:
                    colors[vessel[2]] = vessel[1]

        for k in self.scopes.keys():  # verific corespondenta dintre cantitatile de culoare cu starea finala
            if k not in colors.keys() or self.scopes[k] != colors[k]:
                return False
        return True

    def generate_successors(self, current_node: TraversalNode, heuristic_type: str = "euristica banala") -> list:
        """
        :param current_node: TraversalNode, nodul curent din arborele de parcurgere
        :param heuristic_type: str, tipul euristicii
        :return: list, succesorii nodului dat
        """
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
                litres = min(new_vessels[i][1], new_vessels[j][0] - new_vessels[j][1])  # calculez cantitatea turnata
                if litres == 0:
                    continue
                # actualizez cantitatea de lichid din cele 2 vase
                new_vessels[i] = (new_vessels[i][0], new_vessels[i][1] - litres, new_vessels[i][2])
                new_vessels[j] = (new_vessels[j][0], new_vessels[j][1] + litres, new_vessels[j][2])
                # initial, consider costul = cantitate * costul culorii mutate
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
                        index = None    # verific existenta combinatiei de culori in datele initiale
                        while left <= right:
                            middle = (left + right) // 2
                            if combination == self.combinations[middle][:2]:
                                index = middle
                                break
                            elif combination < self.combinations[middle][:2]:
                                right = middle - 1
                            else:
                                left = middle + 1

                        if index is not None:   # daca exista, actualizez rezultatele si costul
                            new_cost = litres * self.costs[new_vessels[i][2]]
                            new_vessels[j] = (new_vessels[j][0], new_vessels[j][1], self.combinations[index][2])
                        else:   # altfel, din combinatia celor 2 culori rezulta "nedefinit"
                            new_cost = litres * self.costs[new_vessels[i][2]] + \
                                       (new_vessels[j][1] - litres) * self.costs[new_vessels[j][2]]
                            new_vessels[j] = (new_vessels[j][0], new_vessels[j][1], "nedefinit")

                if new_vessels[i][1] == 0:  # daca nu mai am lichid in vasul i, setez culoarea la ""
                    new_vessels[i] = (new_vessels[i][0], new_vessels[i][1], "")

                if not current_node.contains_in_path(new_vessels):
                    # daca nu exista alt nod cu informatia curenta, il adaug in lista de succesori
                    successors_list.append(TraversalNode(new_vessels, current_node, current_node.g + new_cost,
                                                         self.compute_h(new_vessels, heuristic_type)))
        return successors_list

    def compute_h(self, information: list, heuristic_type="euristica banala") -> int:
        """
        :param information: list, informatia nodului curent, pe baza careia se calculeaza h
        :param heuristic_type: str, euristica folosita
        :return: int, valoarea calculata pentru h in functi e de euristica data
        """
        if heuristic_type == "euristica banala":    # daca nodul cu informatia curenta e nod scop, returnez 0, altfel 1
            aux_node = TraversalNode(information, None)
            if self.test_scope(aux_node):
                return 0
            return 1
        elif heuristic_type == "euristica admisibila 1":
            # pentru fiecare culoare, daca nu are fix cantitatea din starea scop, adaug 1, altfel adaug 0
            h = 0
            colors = {}
            for vessel in information:
                if vessel[2] in colors.keys():
                    colors[vessel[2]] = -1
                else:
                    colors[vessel[2]] = vessel[1]

            for vessel in information:
                if vessel[2] not in self.scopes.keys() or self.scopes[vessel[2]] != colors[vessel[2]]:
                    h += 1
            return h
        elif heuristic_type == "euristica admisibila 2":
            # pentru fiecare culoare, adaug cantitatea care mai trebuie obtinuta pentru a ajunge in starea finala
            h = 0
            colors = {}
            for vessel in information:
                if vessel[2] not in colors.keys():
                    colors[vessel[2]] = vessel[1]
                else:
                    colors[vessel[2]] += vessel[1]
            for k in self.scopes.keys():
                if k not in colors.keys():
                    h += self.scopes[k]
                else:
                    h += max(0, self.scopes[k] - colors[k])
            return h
        else:
            # calculez cati litri de apa mai trebuie in total pentru a ajunge la starea finala
            quantity = {}
            for vessel in information:  # cantitatile din fiecare culoare
                if vessel[2] not in quantity.keys():
                    quantity[vessel[2]] = vessel[1]
                else:
                    quantity[vessel[2]] += vessel[1]
            litres = 0
            for q in quantity.keys():
                if q in self.scopes.keys():
                    litres += max(0, self.scopes[q] - quantity[q])
            max_cost = max(self.costs.values())     # calculez costul maxim al unei culori
            # h = numarul de vase * cantitatea totala necesara * costul maxim * 2(se combina 2 culori pentru rezultat) +
            #     costul maxim * cantitatea(se poate muta lichid dintr-un vas in altul)
            h = len(self.start) * litres * max_cost * 2 + max_cost * litres
            return h


def uniform_cost_search(source_graph: Graph, searched_solutions: int) -> None:
    """
    :param source_graph: Graph, graful sursa al problemei
    :param searched_solutions: int, numarul de solutii cautate
    :return: None, afiseaza solutiile gasite
    """
    global output_file, timeout, number_of_solutions

    start_time = time.time()
    queue = [TraversalNode(source_graph.start, None)]  # coada cu noduri
    max_nodes = 1
    total_nodes = 0
    while len(queue) > 0:
        current_node = queue.pop(0)  # extrag primul nod din coada
        if source_graph.test_scope(current_node):  # daca nodul curent este nod scop, afisez solutia
            intermediate_time = time.time()
            output_file.write("----UNIFORM COST SEARCH----\n\nSolutia {}:\n".
                              format(number_of_solutions + 1 - searched_solutions))
            current_node.print_path()
            output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
            output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                              format(max_nodes, total_nodes))
            output_file.write("\n---------------------------------------------------------------------------------\n\n")
            searched_solutions -= 1
            if searched_solutions == 0:
                return

        intermediate_time = time.time()
        if intermediate_time - start_time > timeout:    # cand depasec timeout-ul, ma opresc
            output_file.write("Timpul de executie a depasit timeout-ul!\n")
            break

        successors = source_graph.generate_successors(current_node)     # generez succesorii nodului
        total_nodes += len(successors)
        for s in successors:  # adaug succesorii in coada astfel incat aceasta sa ramana sortata crescator dupa g
            pos = 0
            found_place = False
            for pos in range(len(queue)):   # caut pozitia nodului astfel incat coada sa ramana sortata dupa g
                if queue[pos].g > s.g:
                    found_place = True
                    break
            if found_place:
                queue.insert(pos, s)
            else:
                queue.append(s)
        max_nodes = max(max_nodes, len(queue))


def a_star(source_graph: Graph, searched_solutions: int, heuristic_type="euristica banala") -> None:
    """
    :param source_graph: Graph, graful sursa al problemei
    :param searched_solutions: int, numarul de solutii cautate
    :param heuristic_type: str, euristica folosita
    :return: None, afiseaza solutiile gasite
    """
    global output_file, timeout, number_of_solutions

    start_time = time.time()
    queue = [TraversalNode(source_graph.start, None)]  # coada cu noduri
    max_nodes = 1
    total_nodes = 0
    while len(queue) > 0:
        current_node = queue.pop(0)  # extrag primul nod din coada
        if source_graph.test_scope(current_node):  # daca nodul curent este nod scop, afisez solutia
            intermediate_time = time.time()
            output_file.write("----A*----\n{}\n\nSolutia {}:\n".
                              format(heuristic_type, number_of_solutions + 1 - searched_solutions))
            current_node.print_path()
            output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
            output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                              format(max_nodes, total_nodes))
            output_file.write("\n---------------------------------------------------------------------------------\n\n")
            searched_solutions -= 1
            if searched_solutions == 0:
                return

        intermediate_time = time.time()     # cand timpul de la startul algoritmului depaseste timeoout, oprim cautarea
        if intermediate_time - start_time > timeout:
            output_file.write("Timpul de executie a depasit timeout-ul!\n")
            break

        successors = source_graph.generate_successors(current_node, heuristic_type)
        total_nodes += len(successors)
        for s in successors:  # adaug succesorii in coada astfel incat aceasta sa ramana sortata crescator dupa cost
            pos = 0
            found_place = False
            for pos in range(len(queue)):   # caut pozitia nodului astfel incat coada sa ramana sortata crescator dupa f
                if queue[pos].f >= s.f:
                    found_place = True
                    break
            if found_place:
                queue.insert(pos, s)
            else:
                queue.append(s)
        max_nodes = max(max_nodes, len(queue))


def a_star_optimised(source_graph: Graph, heuristic_type="euristica banala") -> None:
    """
    :param source_graph: Graph, graful sursa al problemei
    :param heuristic_type: str, euristica folosita
    :return: None, afiseaza solutia optima
    """
    global output_file, timeout

    start_time = time.time()
    open_list = [TraversalNode(source_graph.start, None)]   # lista nodurilor de expandat
    closed_list = []    # lista nodurilor expandate
    max_nodes = 1
    total_nodes = 0
    while len(open_list) > 0:
        current_node = open_list.pop(0)     # extrag primul nod din lista de expandat
        closed_list.append(current_node)    # il marchez ca expandat
        if source_graph.test_scope(current_node):   # daca e nod scop, afisez solutia
            intermediate_time = time.time()
            output_file.write("----A* OPTIMISED----\n{}\n\nSolutie:\n".format(heuristic_type))
            current_node.print_path()
            output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
            output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                              format(max_nodes, total_nodes))
            output_file.write("\n---------------------------------------------------------------------------------\n\n")
            return

        intermediate_time = time.time()     # cand depasesc timeout-ul, opresc algoritmul
        if intermediate_time - start_time > timeout:
            output_file.write("Timpul de executie a depasit timeout-ul!\n")
            break

        successors = source_graph.generate_successors(current_node, heuristic_type)
        total_nodes += len(successors)
        for s in successors[:]:
            found_in_open = False
            for node in open_list[:]:   # caut succesorul curent in lista open
                if s.information == node.information:
                    found_in_open = True    # daca il gasesc, il elimin din lista unde are f-ul mai mare
                    if s.f >= node.f:
                        successors.remove(s)
                    else:
                        open_list.remove(node)
            if not found_in_open:   # daca nu este in open, il caut in closed si procedez asemanator
                for node in closed_list[:]:
                    if s.information == node.information:
                        if s.f >= node.f:
                            successors.remove(s)
                        else:
                            closed_list.remove(node)

        for s in successors:    # inserez succesorii in lista open astfel incat sa ramana sortata dupa f
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
        max_nodes = max(max_nodes, len(open_list) + len(closed_list))


def iterative_deepening_a_star(source_graph: Graph, searched_solutions: int, heuristic_type="euristica banala") -> None:
    """
    :param source_graph: Graph, graful sursa al problemei
    :param searched_solutions: int, numarul de solutii cautate
    :param heuristic_type: str, euristica folosita
    :return: None, afiseaza drumurile gasite
    """
    global output_file, timeout

    start_time = time.time()
    start_node = TraversalNode(source_graph.start, None)
    limit = start_node.f
    total_nodes = 0
    while True:
        searched_solutions, result, total_nodes = build_path(source_graph, start_node, limit, searched_solutions,
                                                             start_time, total_nodes, heuristic_type)
        # incerc sa construiesc drumuri cu costul pana la o anumita limita
        intermediate_time = time.time()     # daca algoritmul depaseste timpul, opresc executia
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
               start_time: float, total_nodes: int, heuristic_type="euristica banala") -> tuple:
    """
    :param source_graph: Graph, graful sursa al problemei
    :param current_node: TraversalNode, nodul curent din arborele de parcurgere
    :param limit: int, costul maxim pana la care expandam un drum
    :param searched_solutions: int, numarul de solutii cautate
    :param start_time: float, timpul de start al algoritmului
    :param total_nodes: int, numarul de noduri generate
    :param heuristic_type: str, euristica folosita
    :return: tuple, returneaza numarul de solutii care mai trebuie cautate, un rezultat
            ("done", infinit sau minimul la care s-a ajuns), numarul total de noduri
    """
    global output_file, timeout, number_of_solutions

    if current_node.f > limit:
        return searched_solutions, current_node.f, total_nodes

    max_nodes = 1
    intermediate_time = time.time()     # daca algoritmul depaseste timeout-ul, il opresc
    if intermediate_time - start_time > timeout:
        output_file.write("Timpul de executie a depasit timeout-ul!\n")
        return 0, "done", total_nodes

    max_nodes = max(max_nodes, total_nodes)
    if source_graph.test_scope(current_node) and current_node.f == limit:  # daca este nod scop cu costul egal cu limita
        intermediate_time = time.time()     # afisez solutia
        output_file.write("----ITERATIVE DEEPENING A*----\n{}\n\nSolutia {}:\n".
                          format(heuristic_type, number_of_solutions + 1 - searched_solutions))
        current_node.print_path()
        output_file.write("Timp: " + str(round(1000 * (intermediate_time - start_time)) / 1000) + " secunde\n")
        output_file.write("Numar maxim de noduri existente in memorie: {}\nNumar total de noduri calculate: {}\n".
                          format(max_nodes, total_nodes))
        output_file.write("\n---------------------------------------------------------------------------------\n\n")
        searched_solutions -= 1
        if searched_solutions == 0:
            return 0, "done", total_nodes

    successors = source_graph.generate_successors(current_node, heuristic_type)
    total_nodes += len(successors)
    minimum = float("inf")
    for s in successors:
        # pentru fiecare succesor, incerc sa construiesc un drum
        searched_solutions, result, total_nodes = build_path(source_graph, s, limit, number_of_solutions, start_time,
                                                             total_nodes, heuristic_type)
        if result == "done":
            return 0, "done", total_nodes
        if result < minimum:
            minimum = result
            # print("Noul minim: " + str(minimum))

    return searched_solutions, minimum, total_nodes


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


def compute_quantity(color: str, current: dict, combinations: list) -> int:
    """
    :param color: str, culoarea pentru care calculez cantitatea maxima ce se poate obtine
    :param current: dict, contine culorile existente in vase si cantitatile lor
    :param combinations: list, lista de amestecuri si rezultatele lor
    :return: int, cantitatea maxima
    """
    quantity = 0
    if color in current.keys():
        quantity += current[color]  # initial, cantitatea este cat se gaseste in vase
    left = 0
    right = len(combinations) - 1
    index = None
    while left <= right:    # caut o combinatie care are ca rezultat culoarea curenta
        middle = (left + right) // 2
        if combinations[middle][2] == color:
            index = middle
            break
        elif combinations[middle][2] < color:
            left = middle + 1
        else:
            right = middle - 1

    if index is None:   # daca nu am gasit o combinatie potrivita, returnez ce s-a calculat pana acum
        return quantity
    else:   # altfel, returnez quantity + cantitatile celor 2 culori care dau rezultatul curent
        return quantity + compute_quantity(combinations[index][0], current, combinations) + \
               compute_quantity(combinations[index][1], current, combinations)


def has_scope(current: list, scope: dict, combinations: list) -> bool:
    """
    :param current: list, vasele din starea curenta
    :param scope: dict, culorile din starea scop si cantitatile dorite
    :param combinations: list, lista de amestecuri si rezultatele lor
    :return: bool, True daca exista o stare scop pe drumul care continua de aici, altfel False
    """
    capacities = [c[0] for c in current]    # capacitatile vaselor
    capacities.sort(reverse=True)
    quantities = list(scope.values())       # cantitatile de culoare dorite
    quantities.sort(reverse=True)

    if len(quantities) > len(capacities):   # daca vreau sa obtin mai multe culori decat vase
        return False

    for i in range(len(quantities)):        # daca sunt cantitati care nu pot fi tinute intr-un singur vas
        if i >= len(capacities) or capacities[i] < quantities[i]:
            return False

    colors = {}     # calculez cantitatile existente din fiecare culoare
    for c in current:
        if c[2] != "" and c[2] != "nedefinit":
            if c[2] not in colors.keys():
                colors[c[2]] = c[1]
            else:
                colors[c[2]] += c[1]
    combinations.sort(key=lambda c: c[2])   # sortez lista de combinatii alfabetic dupa rezultat
    for item in scope.items():
        max_quantity = compute_quantity(item[0], colors, combinations)
        if max_quantity < item[1]:  # daca exista vreo culoare pentru care nu pot obtine cantitatea dorita la final
            return False            # returnez False

    return True


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
        valid_data = True
        graph = Graph(file)
        if valid_data:
            uniform_cost_search(graph, number_of_solutions)
            a_star(graph, number_of_solutions, "euristica neadmisibila")
            a_star_optimised(graph, "euristica admisibila 1")
            iterative_deepening_a_star(graph, number_of_solutions, "euristica neadmisibila")
        output_file.close()
