import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pprint import pprint


from api import logic, utils


def main():
    st.header("Application")

    st.subheader("Number 1")

    st.markdown("#### 3x3 Matrices")

    cont_3 = {
        "A": np.array([[0, 1, 0], [3, 0, 1], [0, 1, 0]]),
        "B": np.array([[1], [1], [3]]),
        "C": np.array([0, 1, 0]).reshape(1, -1),
        "D": None,
    }
    c1, c2 = st.columns(2)

    T3 = c1.number_input(
        "T Matrix 3x3", value=0.001, min_value=0.0001, max_value=100.0, format="%f"
    )
    k3 = c2.number_input(
        "k max Matrix 3x3",
        value=1000,
        min_value=1,
    )

    disc_3 = logic.bilinear_transform(cont_3, T3)
    st.write("Matrix 3x3 Discrete State Space", disc_3)

    response_df_3 = logic.step_response_loop(disc_3, k3, T3)
    st.line_chart(response_df_3, x="k", y="Step Response")
    imp_fig_3 = go.Figure()
    utils.add_one_scatter(
        imp_fig_3, response_df_3, x="k", y="Impulse Response", name="Impulse Response"
    )
    st.plotly_chart(imp_fig_3, use_container_width=True)

    st.markdown("#### 2x2 Matrices")

    cont_2 = {
        "A": np.array([[0, 1], [-4, 0]]),
        "B": np.array([[0], [1]]),
        "C": np.array([0, 1]).reshape(1, -1),
        "D": None,
    }

    c3, c4 = st.columns(2)

    T2 = c3.number_input(
        "T Matrix 2x2", value=0.001, min_value=0.0001, max_value=100.0, format="%f"
    )
    k2 = c4.number_input(
        "k max Matrix 2x2",
        value=1000,
        min_value=1,
    )

    disc_2 = logic.bilinear_transform(cont_2, T2)
    st.write("Matrix 2x2 Discrete State Space", disc_2)

    response_df_2 = logic.step_response_loop(disc_2, k2, T2)
    st.line_chart(response_df_2, x="k", y="Step Response")
    imp_fig = go.Figure()
    utils.add_one_scatter(
        imp_fig, response_df_2, x="k", y="Impulse Response", name="Impulse Response"
    )
    st.plotly_chart(imp_fig, use_container_width=True)

    st.subheader("Number 2")

    st.markdown("#### True Systems")
    st.latex(
        r"""
        \frac{d^2y}{dt^2} = -0.5\frac{dy}{dt} - y + u\\
y = x_1 \to \dot x_1 = x_2\\
\frac{dy}{dt} =x_2 \to \dot x_2 = \frac{d^2y}{dt^2}=-0.5x_2-x_1+u\\
 \begin{bmatrix}
\dot x_1 \\
\dot x_2
\end{bmatrix} = \begin{bmatrix}
0 & 1 \\
-1 & -0.5 \\
\end{bmatrix}\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} + 
\begin{bmatrix}
0 \\
1
\end{bmatrix} u\\
y = \begin{bmatrix}
1 & 0 \\
\end{bmatrix} \begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
                """
    )
    cont_2_2 = {
        "A": np.array([[0, 1], [-1, -0.5]]),
        "B": np.array([[0], [1]]),
        "C": np.array([0, 1]).reshape(1, -1),
        "D": None,
    }
    c5, c6 = st.columns(2)

    Th = c5.number_input(
        "T Matrix True Ho-Kalman",
        value=0.1,
        min_value=0.0001,
        max_value=100.0,
        format="%f",
    )
    kh = c6.number_input(
        "k max Matrix True Ho-Kalman",
        value=300,
        min_value=1,
    )

    disc_2_2 = logic.bilinear_transform(cont_2_2, Th)
    st.write("True System State Space in  Discrete", disc_2_2)
    response_df_2_2 = logic.step_response_loop(disc_2_2, kh, Th)
    st.line_chart(response_df_2_2, x="k", y="Step Response")
    imp_fig_3 = go.Figure()
    utils.add_one_scatter(
        imp_fig_3, response_df_2_2, x="k", y="Impulse Response", name="Impulse Response"
    )
    st.plotly_chart(imp_fig_3, use_container_width=True)

    st.markdown("#### Ho-Kalman Estimation")

    c7, c8 = st.columns(2)
    n_input = c7.number_input("Ho-Kalman N data input", value=100, min_value=1)
    n_svd = c8.number_input(
        "Ho-Kalman SVD data",
        value=2,
        min_value=1,
    )

    ho_df = logic.ho_kalman(
        response_df_2_2["Impulse Response"], n_input, n_svd)

    st.write("Ho Kalman Estimation State Space", ho_df)
    response_df_ho_kalman = logic.step_response_loop(ho_df, kh, Th)

    st.line_chart(response_df_ho_kalman, x="k", y="Step Response")
    imp_fig_ho = go.Figure()
    utils.add_one_scatter(
        imp_fig_ho,
        response_df_ho_kalman,
        x="k",
        y="Impulse Response",
        name="Impulse Response",
    )
    st.plotly_chart(imp_fig_ho, use_container_width=True)


if __name__ == "__main__":
    main()
