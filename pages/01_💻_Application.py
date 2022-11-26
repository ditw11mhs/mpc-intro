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

    disc_3 = logic.bilinear_transform(cont_3, 1e-3)
    st.write(disc_3)

    response_df_3 = logic.step_response_loop(disc_3, 1000, 1e-3)
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

    disc_2 = logic.bilinear_transform(cont_2, 1e-3)
    st.write(disc_2)

    response_df_2 = logic.step_response_loop(disc_2, 2000, 1e-3)
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

    disc_2_2 = logic.bilinear_transform(cont_2_2, 0.1)
    st.write("True System State Space in  Discrete", disc_2_2)
    response_df_2_2 = logic.step_response_loop(disc_2_2, 300, 0.1)
    st.line_chart(response_df_2_2, x="k", y="Step Response")
    imp_fig_3 = go.Figure()
    utils.add_one_scatter(
        imp_fig_3, response_df_2_2, x="k", y="Impulse Response", name="Impulse Response"
    )
    st.plotly_chart(imp_fig_3, use_container_width=True)

    st.markdown("#### Ho-Kalman Estimation")

    ho_df = logic.ho_kalman(response_df_2_2["Impulse Response"], 100, 10)
    st.write("Ho Kalman Estimation State Space", ho_df)
    response_df_ho_kalman = logic.step_response_loop(ho_df, 300, 0.1)

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
