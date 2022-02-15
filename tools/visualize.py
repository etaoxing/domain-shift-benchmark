import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import collections

import plotly.graph_objects as go


RECORD_FILE = 'record.csv'
IT_INDEX_LABEL = '_diagnostics/total_iterations'
TASK_TAG = 'eval-d0_task-0-t-microwave,kettle,switch,slide_domainshift'


SHIFT_CATEGORIES = {
    '0': dict(name='training', ignore_stats_in_total=True),
    'change_object': dict(
        name='(a) object instance',
    ),
    'change_object_layout': dict(
        name='(b) object layout',
    ),
    'change_texture': dict(
        name='(c) texture',
    ),
    'change_camera': dict(
        name='(d) camera view',
    ),
    'change_lighting': dict(
        name='(e) lighting',
    ),
    # 'change_noise': {},  # ignore, this was bugged and affecting robot controller
    'change_robot_init': dict(
        name='(f) robot state',
    ),
    'change_one_object_done': dict(
        name='(g) object state',
        # exclude to not inflate perf, since includes
        # setting state of object to done that is involved in the task.
        ignore_stats_in_total=True,
    ),
}
SHIFT_CATEGORIES_LIST = list(SHIFT_CATEGORIES.keys())

RECOMPUTE_TAGS = [
    'success_num_obj/Mean',
    'success_num_obj/Std',
]
for o in ['microwave', 'kettle', 'bottomknob', 'topknob', 'switch', 'slide', 'hinge']:
    RECOMPUTE_TAGS.append(f'eoe/_goaldiff_{o}/Mean')
    RECOMPUTE_TAGS.append(f'eoe/_goaldiff_{o}/Std')


MODELS_CMP_SHIFT = {
    "BC": dict(
        exp='exp_compare_bc',
        method='bc',
    ),
    "BC (MSE)": dict(
        exp='exp_compare_bc',
        method='bc_mse',
    ),
    # "BC w/ inv. dynamics, MSE": dict(
    #     exp='exp_compare_bc',
    #     method='bc_inverse_dynamics',
    #     graph=False,
    # ),
    "BC w/ inv. dynamics": dict(
        exp='exp_compare_bc',
        method='bc_inverse_mixture_dynamics',
    ),
    # "BC w/ contextual LSTM": dict(
    #     exp='exp_compare_bc',
    #     method='bc_contextual_lstm',
    #     graph=False,
    # ),
    # #
    "GCBC (HER relabeling)": dict(
        exp='exp_compare_bc',
        method='gcbc_her',
        graph=False,
    ),
    # "GCBC w/ RPL": dict(
    #     exp='exp_compare_bc',
    #     # method='gcbc_rpl',
    #     method='gcbc_rpl_bstraj4',
    #     graph=False,
    # ),
    # #
    # "BC, online": dict(
    #     exp='exp_compare_finetune',
    #     method='bc_finetune',
    #     graph=False,
    # ),
    # "GCBC w/ RPL, online": dict(
    #     exp='exp_compare_finetune',
    #     method='gcbc_rpl_finetune',
    # ),
    #
    "BC (IMPALA encoder)": dict(
        exp='exp_compare_representation',
        method='impala_encoder',
        graph=False,
    ),
    # "BC (ViT)": dict(
    #     exp='exp_compare_representation',
    #     method='patch_vit',
    # ),
    #
    "BC (β-VAE)": dict(
        exp='exp_compare_representation',
        method='beta_vae',
    ),
    "BC (β-VAE), no stop grad": dict(
        exp='exp_compare_representation',
        method='beta_vae_nodetach',
        graph=False,
    ),
    "BC (RAE)": dict(
        exp='exp_compare_representation',
        method='rae',
        graph=False,
    ),
    "BC (sigma-VAE)": dict(
        exp='exp_compare_representation',
        method='sigma_vae',
        graph=False,
    ),
    # #
    # "BC (SimSiam, random shift)": dict(
    #     exp='exp_compare_representation',
    #     method='sim_siam',
    #     graph=False,
    # ),
    "BC (SimSiam, full aug.)  ": dict(  # additional spaces added so margin on graph legend
        exp='exp_compare_representation',
        method='sim_siam_origaugs',
    ),
    # "BC (SODA, random shift)": dict(
    #     exp='exp_compare_representation',
    #     method='soda',
    #     graph=False,
    # ),
    "BC (ATC, random shift)": dict(
        exp='exp_compare_representation',
        method='time_nce',
    ),
    #
    # "imagenet_resnet18": dict(
    #     exp='exp_compare_representation',
    #     method='imagenet_resnet18',
    # ),
    # "imagenet_resnet18_frozen": dict(
    #     exp='exp_compare_representation',
    #     method='imagenet_resnet18_frozen',
    # ),
    # "imagenet_resnet18_scratch": dict(
    #     exp='exp_compare_representation',
    #     method='imagenet_resnet18_scratch',
    # ),
    # "clip_frozen": dict(
    #     exp='exp_compare_representation',
    #     method='clip_frozen',
    # ),
    #
    "BC (fixed random embedding)    ": dict(  # additional spaces added so margin on graph legend
        exp='exp_compare_representation',
        method='random_embedding',
    ),
    "Random actions": dict(
        exp='exp_compare_baseline',
        method='random_action',
        it=100,
        graph=False,
    ),
    "Demo playback": dict(
        exp='exp_compare_baseline',
        method='demo_playback',
        it=100,
    ),
    "GCBC w/ HER relabeling, more demos": dict(
        exp='exp_compare_moredemos',
        method='gcbc_her_moredemos_startmicrowave',
        it=200000,
        graph=False,
    ),
}


MODELS_CMP_INITPERTURB = {
    "BC, state vector, eval orig": dict(
        exp='exp_compare_initperturb',
        method='bc_statevec_evalorig',
    ),
    "BC, eval orig": dict(
        exp='exp_compare_initperturb',
        method='bc_evalorig',
    ),
    "BC (β-VAE), eval orig": dict(
        exp='exp_compare_initperturb',
        method='bc_beta_vae_evalorig',
    ),
    "Demo playback, eval orig": dict(
        exp='exp_compare_initperturb',
        method='demo_playback_evalorig',
        it=100,
    ),
    #
    "BC, state vector": dict(
        exp='exp_compare_initperturb',
        method='bc_statevec',
    ),
    "BC": MODELS_CMP_SHIFT["BC"],
    "BC (β-VAE)": MODELS_CMP_SHIFT["BC (β-VAE)"],
    "Demo playback": MODELS_CMP_SHIFT["Demo playback"],
}


MODELS = MODELS_CMP_SHIFT
# MODELS = {"BC": MODELS_CMP_SHIFT["BC"], "BC (β-VAE)": MODELS_CMP_SHIFT["BC (β-VAE)"]}
# MODELS = MODELS_CMP_INITPERTURB


def recompute_stats(r):
    # Compute stats across categories instead of individual domain shifts
    s = collections.defaultdict(list)
    for k, v_params in SHIFT_CATEGORIES.items():
        if v_params.get('ignore_stats_in_total', False):
            continue

        for t in RECOMPUTE_TAGS:
            _d = r[f'{TASK_TAG}-{k}/{t}']

            s[t].append(_d)

    for t, v in s.items():
        r[f'{TASK_TAG}-all/{t}'] = np.mean(v)

    m0 = r[f'{TASK_TAG}-0/success_num_obj/Mean']
    ma = r[f'{TASK_TAG}-all/success_num_obj/Mean']
    r[f'{TASK_TAG}-all/success_num_obj/RelDiff'] = (ma - m0) / m0
    return r


def compute_total_runtime(rs, num_seeds):
    s = 0
    for k, row in rs.items():
        try:
            t = row['_diagnostics/elapsed_time_sec']
            s += t * num_seeds
            print(f'average time per run for {k}: {t}')
        except Exception as e:
            pass

    print(f'total runtime: {s}')
    print(f'average runtime per method: {s / len(rs)}')


def visualize(args):
    experiment_dir = args.d

    seeds = []

    rps = {}

    for n, v in MODELS.items():
        record_paths = sorted(
            glob(
                os.path.join(
                    experiment_dir,
                    v['exp'],
                    v['method'],
                )
                + f'/*/{RECORD_FILE}'
            )
        )
        rps[n] = record_paths
        seeds += [x.split('/')[-2].split('seed')[-1] for x in record_paths]

    seeds = set(seeds)

    print(seeds)

    rs = {}
    for n, v in MODELS.items():

        rows = []

        for record_path in rps[n]:
            print(record_path)
            it = v.get('it', args.it)
            try:
                df = pd.read_csv(record_path)
                row = df.loc[df[IT_INDEX_LABEL] == it].iloc[0]
            except Exception as e:
                print(f'skipped {record_path}, {e}')
                continue

            row = recompute_stats(row)
            rows.append(row)

        df = pd.DataFrame(rows)

        r = df.mean()
        rs[n] = r

    compute_total_runtime(rs, len(seeds))

    print('\n---\n')

    with pd.option_context('display.float_format', '{:0.2f}'.format):

        for k, v in rs.items():
            print(k)
            print(
                v[
                    [
                        f'{TASK_TAG}-0/success_num_obj/Mean',
                        f'{TASK_TAG}-0/success_num_obj/Std',
                        f'{TASK_TAG}-all/success_num_obj/Mean',
                        f'{TASK_TAG}-all/success_num_obj/Std',
                        f'{TASK_TAG}-all/success_num_obj/RelDiff',
                    ]
                ],
            )
            print()

    if args.gp is not None:
        _graph_by_category(args, {k: v for k, v in rs.items() if MODELS[k].get('graph', True)})


def _graph_by_category(args, rs):
    data = []

    for m, v in rs.items():
        mean = []
        std = []
        for k in SHIFT_CATEGORIES_LIST:
            mean.append(v[f'{TASK_TAG}-{k}/success_num_obj/Mean'])
            std.append(v[f'{TASK_TAG}-{k}/success_num_obj/Std'])

        b = go.Bar(
            name=m,
            x=[SHIFT_CATEGORIES[k]['name'] for k in SHIFT_CATEGORIES_LIST],
            y=mean,
            error_y=dict(
                type='data',
                array=std,
                #
                width=0,
                # comment if want to only graph +/- std
                symmetric=False,
                arrayminus=[0] * len(std),
            ),
        )

        data.append(b)

    fig = go.Figure(data)
    fig.update_layout(
        barmode='group',
        #
        xaxis=dict(
            title='<b>Domain shift category</b>',
            titlefont=dict(size=30),
        ),
        yaxis=dict(
            title='<b>Avg. steps completed</b>',
            titlefont=dict(size=30),
            gridcolor='rgb(230,236,245)',
        ),
        width=1200,
        height=618,
        font=dict(family="Open Sans", size=24, color='black'),
        plot_bgcolor='rgb(255,255,255)'
        #
    )
    fig.update_yaxes(range=[0, 4.5], tickvals=list(np.arange(0, 4.5, 0.5)))

    # fig.show()
    # fig.write_image(args.gp)

    _fig_to_server(fig)


def _fig_to_server(fig):
    # need to use Dash b/c local fonts not working...

    import io
    from base64 import b64encode

    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    import plotly.express as px

    buffer = io.StringIO()

    fig.write_html(buffer)

    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            html.Link(
                rel="preconnect",
                href="https://fonts.googleapis.com",
            ),
            html.Link(
                rel="preconnect",
                href="https://fonts.gstatic.com",
            ),
            html.Link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,300;1,400;1,600;1,700;1,800&display=swap",
            ),
            # html.A(
            #     html.Button("Download HTML"),
            #     id="download",
            #     href="data:text/html;base64," + encoded,
            #     download="plotly_graph.html",
            # ),
            dcc.Graph(id="graph", figure=fig, config={'displayModeBar': False}),
        ]
    )

    app.run_server(debug=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='experiment dir')
    parser.add_argument('-it', help='eval iteration', type=int)
    parser.add_argument('-gp', default=None, help='output graph path', type=str)
    args = parser.parse_args()
    visualize(args)


# python tools/visualize.py -d ~/workdir_experiments/domain_shift_benchmark/ -it 100000 -gp ~/gcb_figures/figure_domain_shift_categories.pdf
