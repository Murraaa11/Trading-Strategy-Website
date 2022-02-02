import plotly.graph_objects as go


def add_layout(fig, title, x_title, y_title):

    xformat = dict(rangeslider=dict(visible=True), type="date")

    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'font': {'size': 22}
        },
        paper_bgcolor='white',
        plot_bgcolor='white',
        autosize=False,
        height=450,
        xaxis={**xformat, **{
            'title': x_title,
            'showline': True,
            'linewidth': 1,
            'linecolor': 'black',
        }},
        yaxis={
            'title': y_title,
            'showline': True,
            'linewidth': 1,
            'linecolor': 'black'
        },
        legend={
            'orientation': 'h',  # 图例横着排列还是竖着排列：h stands for horizontal
            # This orients the legend but it will result in an overlay of the graph
            # 'yanchor': 'top', 'y': 0.5,
            'yanchor': 'top', 'y': 1.15,  # 'top' 指上端在整个图里位置的比例（可以在0-1范围之外）
            'xanchor': 'right', 'x': 1,  # ‘right’指右端在整个图位置的比例
        },
        # this fixes the overlay
        margin={'t': 100},
    )
    fig.layout.hovermode = 'x'
    # fig.update_traces(visible = 'legendonly')

    return (fig)