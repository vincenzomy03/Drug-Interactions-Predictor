class Var:
    """Rappresenta una variabile logica in Prolog (es. ?X o ?Role)."""
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"?{self.name}"

    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


def walk(term, subst):
    """Risolve ricorsivamente le variabili associate nel dizionario di sostituzione."""
    while isinstance(term, Var) and term in subst:
        term = subst[term]
    return term


def unify(x, y, subst):
    """Esegue l'unificazione logica tra due termini x e y data una sostituzione."""
    if subst is None:
        return None
    x = walk(x, subst)
    y = walk(y, subst)

    if x == y:
        return subst
    if isinstance(x, Var):
        return {**subst, x: y}
    if isinstance(y, Var):
        return {**subst, y: x}
    if isinstance(x, tuple) and isinstance(y, tuple):
        if len(x) != len(y):
            return None
        new_subst = subst
        for xi, yi in zip(x, y):
            new_subst = unify(xi, yi, new_subst)
            if new_subst is None:
                return None
        return new_subst
    return None


def rename_vars(term, suffix):
    """Rinomina le variabili in un termine per evitare collisioni di scope durante il backtracking."""
    if isinstance(term, Var):
        return Var(f"{term.name}_{suffix}")
    if isinstance(term, tuple):
        return tuple(rename_vars(x, suffix) for x in term)
    return term


class Prolog:
    """Motore inferenziale Prolog per la Programmazione Logica in Python."""
    def __init__(self):
        self.clauses = []
        self.var_counter = 0

    def add_fact(self, term):
        """Aggiunge un fatto alla Knowledge Base."""
        self.clauses.append((term, []))

    def add_rule(self, head, body):
        """Aggiunge una regola alla Knowledge Base nella forma: head :- body."""
        self.clauses.append((head, body))

    def query(self, *goals):
        """Esegue una query logica e restituisce un generatore di sostituzioni valide."""
        self.var_counter = 0
        yield from self._solve(list(goals), {})

    def _solve(self, goals, subst):
        if not goals:
            yield subst
            return

        goal = goals[0]
        resolved_goal = self._resolve_term(goal, subst)

        for head, body in self.clauses:
            self.var_counter += 1
            suffix = str(self.var_counter)
            r_head = rename_vars(head, suffix)
            r_body = [rename_vars(b, suffix) for b in body]

            new_subst = unify(resolved_goal, r_head, subst)
            if new_subst is not None:
                new_goals = r_body + goals[1:]
                yield from self._solve(new_goals, new_subst)

    def _resolve_term(self, term, subst):
        term = walk(term, subst)
        if isinstance(term, tuple):
            return tuple(self._resolve_term(x, subst) for x in term)
        return term
