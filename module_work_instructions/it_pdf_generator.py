from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

class it_PDFGenerator:
    """
        Gerador Profissional de PDF no padrão industrial com suporte a Imagens Ilustrativas.
    """

    def __init__(self, data: dict, output_path: str):
        self.data = data
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='TableText',
            fontSize=8,
            leading=10,
            alignment=0
        ))
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            fontSize=9,
            leading=11,
            alignment=1,
            fontName='Helvetica-Bold'
        ))
        # Estilo específico para o objetivo com quebra de linha
        self.styles.add(ParagraphStyle(
            name='ObjectiveText',
            fontSize=8,
            leading=9,
            alignment=0
        ))

    def generate(self):
        data = self.data

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont('Helvetica', 8)
            canvas.drawString(doc.leftMargin, 0.75 * cm, "Powered by NewGen Intelligent Engineering Solutions (newgensolutions.com.br)")
            canvas.restoreState()

        doc = SimpleDocTemplate(self.output_path, pagesize=landscape(A4),
                                rightMargin=0.8 * cm, leftMargin=0.8 * cm,
                                topMargin=0.8*cm, bottomMargin=0.8*cm)
        elements = []

        # --- CABEÇALHO ---
        # Transformando campos em Parágrafos para suportar quebra de linha automática
        titulo_p = Paragraph(f"<b>Instrução de Trabalho - {data['titulo']}</b>", self.styles['TableHeader'])
        local_p = Paragraph(f"<b>LOCAL:</b> {data['local']}", self.styles['TableText'])
        objetivo_p = Paragraph(f"<b>OBJETIVO:</b> {data['objetivo']}", self.styles['ObjectiveText'])
        autor_p = Paragraph(f"<b>AUTOR:</b> {data.get('autor', 'NewGen/OptiGen AI Agent')}", self.styles['TableText'])

        header_data = [
            ['EMPRESA', titulo_p, ''],
            [f'Nº DA IT: {data.get("id", "NOVO")}', local_p, objetivo_p],
            [f'VERSÃO: {data.get("versao", "001")}', autor_p, f'APROVADOR: Pendente']
        ]

        # Ajuste fino das larguras para evitar atropelos (Total 28cm para landscape A4)
        header_table = Table(header_data, colWidths=[4*cm, 11*cm, 13*cm])
        header_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('SPAN', (1,0), (2,0)),
            ('BACKGROUND', (0,0), (0,0), colors.lightgrey),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.4*cm))

        # --- MATRIZ DE SEGURANÇA ---
        safety = data['matriz_seguranca']
        safety_data = [
            [Paragraph('RISCOS', self.styles['TableHeader']),
             Paragraph('CONTROLES CRÍTICOS', self.styles['TableHeader']),
             Paragraph('CRITÉRIOS DE PARADA', self.styles['TableHeader']),
             Paragraph('EQUIPAMENTOS / EPIS', self.styles['TableHeader'])]
        ]

        riscos_p = Paragraph("<br/>".join([f"• {r}" for r in safety['riscos']]), self.styles['TableText'])
        controles_p = Paragraph("<br/>".join([f"• {c}" for c in safety['controles_criticos']]), self.styles['TableText'])
        parada_p = Paragraph("<br/>".join([f"• {p}" for p in safety['criterios_parada']]), self.styles['TableText'])
        lista_equip = safety.get('equipamentos_ferramentas', []) + safety.get('epis', [])
        equip_p = Paragraph("<br/>".join([f"• {e}" for e in lista_equip]), self.styles['TableText'])

        safety_data.append([riscos_p, controles_p, parada_p, equip_p])

        safety_table = Table(safety_data, colWidths=[7*cm, 7*cm, 7*cm, 7*cm])
        safety_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,0), colors.red),
            ('BACKGROUND', (1,0), (1,0), colors.green),
            ('BACKGROUND', (2,0), (2,0), colors.orange),
            ('BACKGROUND', (3,0), (3,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (2,0), colors.white),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        elements.append(safety_table)
        elements.append(Spacer(1, 0.4*cm))

        # --- FLUXO DE EXECUÇÃO COM ESPAÇO PARA FOTO ---
        # Adicionamos a coluna "FOTO" (IMAGEM ILUSTRATIVA)
        steps_header = [['PASSO', 'O QUE FAZER?', 'COMO FAZER?', 'SEGURANÇA', 'FOTO ILUSTRATIVA']]
        # Ajustamos as larguras para acomodar a foto
        exec_widths = [1.5*cm, 6*cm, 9.5*cm, 6*cm, 5*cm]
        steps_table_header = Table(steps_header, colWidths=exec_widths)
        steps_table_header.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.navy),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ]))
        elements.append(steps_table_header)

        for step in data['fluxo_execucao']:
            # Espaço vazio na última coluna para colar a foto
            step_data = [[
                str(step['passo_n']),
                Paragraph(step['o_que_fazer'], self.styles['TableText']),
                Paragraph(f"{step['como_fazer']}<br/><br/><b>Por que:</b> {step['por_que_fazer']}", self.styles['TableText']),
                Paragraph("<br/>".join(step['medidas_controle']), self.styles['TableText']),
                "" # Espaço para foto
            ]]
            st = Table(step_data, colWidths=exec_widths)
            st.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('MINHEIGHT', (0,0), (-1,-1), 2.5*cm), # Garante espaço mínimo para a foto
            ]))
            elements.append(st)

        doc.build(elements, onFirstPage=footer, onLaterPages=footer)
        return self.output_path

if __name__ == "__main__":
    dados_exemplo = {
        "titulo": "Operação de Ponte Rolante ECL",
        "local": "Redução sala de cubas L-3",
        "objetivo": "Garantir a integridade física dos colaboradores durante a troca periódica dos ânodos e manutenção preventiva do sistema de exaustão das cubas eletrolíticas.",
        "matriz_seguranca": {
            "riscos": ["Batida por carga suspensa", "Choque elétrico"],
            "controles_criticos": ["Isolamento da área", "Inspeção pré-operacional"],
            "criterios_parada": ["Buzina inoperante"],
            "equipamentos_ferramentas": ["Extrator de ânodos"],
            "epis": ["Capuz Retardante"]
        },
        "fluxo_execucao": [
            {
                "passo_n": 1,
                "o_que_fazer": "Acessar a garagem",
                "como_fazer": "Usar escada com corrimão, mantendo 3 pontos de contato.",
                "por_que_fazer": "Evitar quedas de nível.",
                "medidas_controle": ["Uso de bota antiderrapante"]
            }
        ]
    }
    gen = it_PDFGenerator(dados_exemplo, "IT_Empresa_Melhorada.pdf")
    gen.generate()
    print("PDF Melhorado gerado com sucesso!")
