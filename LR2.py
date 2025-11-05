import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Problem Definition --------------------
class GAProblem:
    def __init__(self, name, dim, fitness_fn):
        self.name = name
        self.dim = dim
        self.fitness_fn = fitness_fn

# Custom fitness: dynamically uses target_ones and max_fitness
def make_fitness(max_fitness: float, target_ones: int):
    def fitness(x: np.ndarray) -> float:
        n_ones = int(np.sum(x))
        # scaled penalty so fitness is max_fitness when n_ones == target_ones
        # and decreases linearly with difference
        penalty_per_diff = max_fitness / target_ones if target_ones != 0 else 0.0
        return float(max_fitness - abs(target_ones - n_ones) * penalty_per_diff)
    return fitness

# -------------------- GA Operators --------------------
def init_population(pop_size: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, dim), dtype=np.int8)

def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    # choose k indices and return index of best among them
    idxs = rng.integers(0, fitness.size, size=k)
    return int(idxs[np.argmax(fitness[idxs])])

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    if a.size <= 1:
        return a.copy(), b.copy()
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def bit_mutation(x: np.ndarray, mut_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def evaluate(pop: np.ndarray, fitness_fn) -> np.ndarray:
    return np.array([fitness_fn(ind) for ind in pop], dtype=float)

# -------------------- GA Run --------------------
def run_ga_streamlit(pop_size, generations, chromosome_length, crossover_rate, mutation_rate, tournament_k, max_fitness, target_ones, seed=42):
    rng = np.random.default_rng(seed)
    
    # Define problem with user-selected chromosome length and target ones
    problem = GAProblem(
        name="Custom Bit Problem", 
        dim=chromosome_length, 
        fitness_fn=make_fitness(max_fitness, target_ones)
    )
    
    pop = init_population(pop_size, problem.dim, rng)
    fit = evaluate(pop, problem.fitness_fn)

    history_best, history_avg, history_worst = [], [], []

    chart_area = st.empty()
    best_area = st.empty()

    for gen in range(generations):
        next_pop = []
        while len(next_pop) < pop_size:
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)
            p1, p2 = pop[i1], pop[i2]

            # Crossover
            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c2)

        pop = np.array(next_pop)
        fit = evaluate(pop, problem.fitness_fn)

        # Track history
        best_idx = int(np.argmax(fit))
        best_fit = float(fit[best_idx])
        avg_fit = float(np.mean(fit))
        worst_fit = float(np.min(fit))
        history_best.append(best_fit)
        history_avg.append(avg_fit)
        history_worst.append(worst_fit)

        # Streamlit live chart update
        df = pd.DataFrame({
            "Best": history_best,
            "Average": history_avg,
            "Worst": history_worst
        })
        chart_area.line_chart(df)
        best_area.markdown(
            f"Generation {gen+1}/{generations} — Best fitness: *{best_fit:.2f}*, Ones: {int(np.sum(pop[best_idx]))}"
        )

    # Return best solution
    best_idx = int(np.argmax(fit))
    best = pop[best_idx]
    return best, float(fit[best_idx]), history_best, history_avg, history_worst

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Genetic Algorithm - Bit Pattern", page_icon="🧬", layout="wide")
st.title("Genetic Algorithm: Generate Bit Pattern")

st.sidebar.header("GA Parameters")

# ----- Sliders for dynamic input -----
pop_size = st.sidebar.slider("Population Size", min_value=50, max_value=1000, value=300, step=10)
chromosome_length = st.sidebar.slider("Chromosome Length", min_value=20, max_value=200, value=80, step=1)
generations = st.sidebar.slider("Number of Generations", min_value=10, max_value=200, value=50, step=1)
max_fitness = st.sidebar.slider("Max Fitness", min_value=10, max_value=200, value=80, step=1)
target_ones = st.sidebar.slider("Target Number of Ones", min_value=1, max_value=200, value=50, step=1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.9, 0.05)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 0.1, 0.01, 0.005)
tournament_k = st.sidebar.slider("Tournament Size", 2, 10, 3)

# Display current parameters dynamically
st.caption(
    f"Population={pop_size}, Chromosome Length={chromosome_length}, "
    f"Max Fitness={max_fitness} at {target_ones} ones, Generations={generations}"
)

if st.button("Run GA"):
    with st.spinner("Running genetic algorithm..."):
        best_solution, best_fitness, hist_best, hist_avg, hist_worst = run_ga_streamlit(
            pop_size=pop_size,
            generations=generations,
            chromosome_length=chromosome_length,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_k=tournament_k,
            max_fitness=max_fitness,
            target_ones=target_ones
        )

    bitstring = ''.join(map(str, best_solution.tolist()))
    st.subheader("Best Solution Found")
    st.code(bitstring, language="text")
    st.write("Best Fitness:", best_fitness)
    st.write("Number of Ones:", int(np.sum(best_solution)))
