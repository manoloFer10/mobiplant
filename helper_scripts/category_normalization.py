import pandas as pd
import numpy as np

plant_species_mapping = {
    'Arabidopsis thaliana': 'Arabidopsis thaliana',
    np.nan: 'non-specific',
    'Nicotiana benthamiana': 'Nicotiana benthamiana',
    'common bean (Phaseolus vulgaris)': 'Phaseolus vulgaris',
    'soybean (Glycine max)': 'Glycine max',
    'Helianthus annuus': 'Helianthus annuus',
    'Pepper': 'Capsicum annuum',
    'Solanum tuberosum': 'Solanum tuberosum',
    'Medicago sativa': 'Medicago sativa',
    'Lactuca sativa': 'Lactuca sativa',
    'Oryza sativa': 'Oryza sativa',
    'Solanum tuberosum, soybean and chrysanthemum': ['Solanum tuberosum', 'Glycine max', 'Chrysanthemum'],
    'Crop': 'non-specific',
    'Arabidopsis thaliana, Physcomitrella patens': ['Arabidopsis thaliana', 'Physcomitrella patens'],
    'Barley and Arabidopsis thaliana': ['Hordeum vulgare', 'Arabidopsis thaliana'],
    'Plant species with drupes': 'non-specific',
    'Plant species with climacteric fruit': 'non-specific',
    'Prunus persica': 'Prunus persica',
    'Plant species with fleshy fruit': 'non-specific',
    'Medicago truncatula': 'Medicago truncatula',
    'Arabidopsis thaliana and Solanum lycopersicum': ['Arabidopsis thaliana', 'Solanum lycopersicum'],
    'This was most extensively studied in Arabidopsis but most probably also holds true foe Marchantia, hence presumably widely applicable.': ['Arabidopsis thaliana', 'Marchantia polymorpha'],
    'Mostly determined in Arabidopsis': 'Arabidopsis thaliana',
    'Mostly studied in Arabidopsis but phototropins as light activated protein kinases was shown in several angiosperms': 'Arabidopsis thaliana',
    'Lotus japonicus': 'Lotus japonicus',
    'Pea': 'Pisum sativum',
    'Solanum lycopersicum': 'Solanum lycopersicum',
    'Jasmonic acid (JA)': 'non-specific',
    'Marchantia polymorpha': 'Marchantia polymorpha',
    'No': 'non-specific',
    'Strawberry': 'Fragaria × ananassa',
    'strawberry': 'Fragaria × ananassa',
    'Arabidopsis thaliana, Hordeum vulgare': ['Arabidopsis thaliana', 'Hordeum vulgare'],
    'Arabidopsis thaliana, Nicotiana benthamiana': ['Arabidopsis thaliana', 'Nicotiana benthamiana'],
    'Arabidopsis thaliana, Brasicaceae family': 'non-specific',
    'Hordeum vulgare': 'Hordeum vulgare',
    'Grasses (Poaceae).': 'non-specific',
    'Arabidopsis thaliana.': 'Arabidopsis thaliana',
    'Angiosperms in general': 'non-specific',
    'Sorghum bicolor': 'Sorghum bicolor',
    'Citrus spp.': 'non-specific',
    'wheat': 'Triticum aestivum',
    'Solanum lycopersicum and Arabidopsis thaliana': ['Solanum lycopersicum', 'Arabidopsis thaliana'],
    'Zea mays': 'Zea mays',
    'Medicago truncatula and Sinorhizobium meliloti': 'Medicago truncatula',
    'Arachis hypogaea': 'Arachis hypogaea',
    'Solanum lycopersicum and other Solanaceae': 'Solanum lycopersicum',
    'Arabidopsis thaliana, Legumes': 'non-specific',
    'Medicago, Arabidopsis': 'non-specific',
    'Legumes': 'non-specific',
    'Brassica spp.': 'non-specific',
    'wheat and barley': ['Triticum aestivum', 'Hordeum vulgare'],
    'Nothofagus pumilio, Arabidopsis thaliana, Populus tomentosa': ['Nothofagus pumilio', 'Arabidopsis thaliana', 'Populus tomentosa'],
    'Nothofagus pumilio': 'Nothofagus pumilio',
    'Populus trichocarpa': 'Populus trichocarpa',
    'It is general for tree species. I think it is useful to test some general question about trees, for people that works, for instance in annual spp and wants to know about something in general for other taxons (i.e. trees). I have an alternative question if this is not suitable.': 'non-specific',
    'Marchantia': 'non-specific',
    'Soybean (Glycine max L.)': 'Glycine max',
    'Parasponia andersonii': 'Parasponia andersonii',
    'Soybean (Glycine max)': 'Glycine max',
    'Glycine max': 'Glycine max',
    'Medicago truncatula and Lotus japonicus': ['Medicago truncatula', 'Lotus japonicus'],
    'Rosidae': 'non-specific',
    'Hevea': 'non-specific',
    'Asteraceae': 'non-specific',
    'Grasses': 'non-specific',
    'Arabidopsis thaliana, Nicotiana benthamiana.': ['Arabidopsis thaliana', 'Nicotiana benthamiana'],
    'Nicotiana benthamiana.': 'Nicotiana benthamiana',
    'Peanut': 'Arachis hypogaea',
    'Arabidosis thaliana': 'Arabidopsis thaliana',
    'Marchantia polymorpha, Arabidopsis thaliana': ['Marchantia polymorpha', 'Arabidopsis thaliana'],
    'Physcomitrella patens, Arabidopsis thaliana': ['Physcomitrella patens', 'Arabidopsis thaliana'],
    'Antirrhinum majus': 'Antirrhinum majus',
    'Arabidopsis, Medicago truncatula': 'non-specific',
    'Arabidopsis thaliana and Oryza sativa': ['Arabidopsis thaliana', 'Oryza sativa'],
    'Citrus trifoliata L': 'Citrus trifoliata',
    'Wheat': 'Triticum aestivum',
    'x': 'non-specific',
    'Welwitschia mirabilis': 'Welwitschia mirabilis',
    'Any CAM plant fit in this section': 'non-specific',
    'Solanum lycopersicum L.': 'Solanum lycopersicum'
}

area_mapping = {
    # Group all GENE REGULATION entries under "Gene Regulation"
    'GENE REGULATION - TRANSCRIPTION': 'GENE REGULATION',
    'GENE REGULATION - POST-TRANSLATIONAL MODIFICATIONS': 'GENE REGULATION',
    'GENE REGULATION - PTGS': 'GENE REGULATION',
    'GENE REGULATION - ALTERNATIVE SPLICING': 'GENE REGULATION',
    'GENE REGULATION - TRANSLATION': 'GENE REGULATION',
    'GENE REGULATION - EPIGENETICS AND TGS': 'GENE REGULATION',
    'GENE REGULATION - EPITRANSCRIPTOMICS AND RNA STRUCTURE': 'GENE REGULATION',
    
    # Group all ENVIRONMENT entries under "Environments"
    'ENVIRONMENT - LIGHT AND TEMPERATURE': 'ENVIRONMENT',
    'ENVIRONMENT - PLANT-SYMBIONTS': 'ENVIRONMENT',
    'ENVIRONMENT - BIOTIC STRESS': 'ENVIRONMENT',
    'ENVIRONMENT - ABIOTIC STRESS': 'ENVIRONMENT',
    'ENVIRONMENT - NUTRIENTS': 'ENVIRONMENT',
    
    # Keep other categories unchanged
    'PLANT BIOTECHNOLOGY': 'PLANT BIOTECHNOLOGY',
    'HORMONES': 'HORMONES',
    'GENOME AND GENOMICS': 'GENOME AND GENOMICS',
    'GROWTH AND DEVELOPMENT': 'GROWTH AND DEVELOPMENT',
    'PHYSIOLOGY AND METABOLISM': 'PHYSIOLOGY AND METABOLISM',
    'EVOLUTION': 'EVOLUTION',
    'CELL BIOLOGY AND CELL SIGNALING': 'CELL BIOLOGY AND CELL SIGNALING'
    }

source_mapping = {
    # Standard URL to DOI mappings
    'https://academic.oup.com/nar/article/51/19/10719/7275011': 'https://doi.org/10.1093/nar/gkad747',
    'https://academic.oup.com/nar/article/48/11/6234/5836195': 'https://doi.org/10.1093/nar/gkaa343',
    'https://www.nature.com/articles/s41477-024-01814-9': 'https://doi.org/10.1038/s41477-024-01814-9',
    'https://elifesciences.org/articles/26770': 'https://doi.org/10.7554/eLife.26770',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC5933142/': 'https://doi.org/10.1104/pp.17.01417',
    'What is the speed of alternative splicing response to cold stress in Arabidopsis leaves?': 'https://doi.org/10.1105/tpc.18.00177',
    'https://www.genome.gov/genetics-glossary/Intron': 'non-specific',
    'https://www.sciencedirect.com/science/article/pii/S1097276514002779?via%3Dihub': 'https://doi.org/10.1016/j.molcel.2014.03.044',
    'https://www.sciencedirect.com/science/article/pii/S1360138519300457?via%3Dihub': 'https://doi.org/10.1016/j.tplants.2019.02.006',
    'ISBN: 978-1-118-50219-8': 'non-specific',
    'https://www.nature.com/articles/s41589-018-0033-4': 'https://doi.org/10.1038/s41589-018-0033-4',
    'https://www.nature.com/articles/s41477-020-0604-8': 'https://doi.org/10.1038/s41477-020-0604-8',
    'https://pubmed.ncbi.nlm.nih.gov/17971042/': 'https://doi.org/10.1111/j.1365-313X.2007.03318.x',
    'https://pubmed.ncbi.nlm.nih.gov/32480511/': 'https://doi.org/10.1071/FP16060',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC58947/': 'https://doi.org/10.1104/pp.122.4.1129',
    'http://www.biomedcentral.com/1471-2229/8/7': 'https://doi.org/10.1186/1471-2229-8-7',
    'www.pnas.org/cgi/doi/10.1073pnas.0906131106': 'https://doi.org/10.1073/pnas.0906131106',
    'https://doi.org/10.3390/horticulturae8111000': 'https://doi.org/10.3390/horticulturae8111000',
    
    # Multi-source entries
    'multiple sources': 'non-specific',
    'mulitple papers required': 'non-specific',
    'multiple': 'non-specific',
    'mulitple': 'non-specific',
    
    # Complex citations
    'Roy, Sonali, et al. \"The peptide GOLVEN10 alters root development and noduletaxis in Medicago truncatula.\" The Plant Journal 118.3 (2024): 607-625., Fernandez, Ana, Pierre Hilson, and Tom Beeckman. \"GOLVEN peptides as important regulatory signalling molecules of plant development.\" Journal of experimental botany 64.17 (2013): 5263-5268., ': 'https://doi.org/10.1111/tpj.16626',
    'Bürger, Marco, and Joanne Chory. "The many models of strigolactone signaling." Trends in plant science 25.4 (2020): 395-405.': 'https://doi.org/10.1016/j.tplants.2019.12.009',
    
    # Special cases
    '…https://www.nature.com/articles/s41598-020-70315-4….': 'https://doi.org/10.1038/s41598-020-70315-4',
    'https://www.nature.com/articles/s41598-020-70315-4': 'https://doi.org/10.1038/s41598-020-70315-4',
    'https://doi.org/10.1371/ journal.pone.0246615': 'https://doi.org/10.1371/journal.pone.0246615',
    'https://www.nature.com/articles/s41467-018-02831-x#Sec2': 'https://doi.org/10.1038/s41467-018-02831-x',
    'https://www.cell.com/molecular-plant/fulltext/S1674-2052(20)30446-9?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS1674205220304469%3Fshowall%3Dtrue': 'https://doi.org/10.1016/j.molp.2020.12.015',
    'https://www.nature.com/articles/nature00996': 'https://doi.org/10.1038/nature00996',
    'https://www.cell.com/molecular-cell/fulltext/S1097-2765(18)30007-8': 'https://doi.org/10.1016/j.molcel.2018.01.007',
    'https://academic.oup.com/plphys/article/180/1/392/6117647?login=true': 'https://doi.org/10.1104/pp.18.01106',
    'https://pubmed.ncbi.nlm.nih.gov/38254210/': 'https://doi.org/10.1186/s13059-024-03163-4',
    'https://www.nature.com/articles/s41467-019-13045-0': 'https://doi.org/10.1038/s41467-019-13045-0',
    'https://www.nature.com/articles/nature02039': 'https://doi.org/10.1038/nature02039',
    'https://www.nature.com/articles/ncomms14534': 'https://doi.org/10.1038/ncomms14534',
    'https://www.nature.com/articles/ncomms2621': 'https://doi.org/10.1038/ncomms2621',
    'https://www.mdpi.com/2079-7737/11/6/861': 'https://doi.org/10.3390/biology11060861',
    'https://www.mdpi.com/2311-7524/8/11/1000': 'https://doi.org/10.3390/horticulturae8111000',
    'https://pubmed.ncbi.nlm.nih.gov/36137053/ https://pubmed.ncbi.nlm.nih.gov/19766570/': 'https://doi.org/10.1126/science.add1104',
    'https://pubmed.ncbi.nlm.nih.gov/19766570/': 'https://doi.org/10.1016/j.cell.2009.07.004',
    'https://pubmed.ncbi.nlm.nih.gov/24942915/': 'https://doi.org/10.1093/jxb/eru245',
    'Montez, M.; Majchrowska, M.; Krzyszton, M.; Bokota, G.; Sacharowski, S.; Wrona, M.; Yatusevich, R.; Massana, F.; Plewczynski, D.; Swiezewski, S. Promoter-pervasive transcription causes RNA polymerase II pausing to boost DOG1 expression in response to salt. EMBO J. 2023, 42, e112443.': 'https://doi.org/10.15252/embj.2022112443',
    'x': 'https://doi.org/10.1093/jxb/eru261',
    'https://doi.org/10.1146/annurev-arplant-050718- 095946': 'https://doi.org/10.1146/annurev-arplant-050718-095946',
    '10.1146/annurev-arplant-050718-': 'DOI: 10.1146/annurev-arplant-050718-100005',
    'https://doi.org/10.1007/s13199-024-01022-1nutrient.': 'https://doi.org/10.1007/s13199-024-01022-1',
    'https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2019.01779/full#B36':'https://doi.org/10.3389/fpls.2019.01779',
    '10.1111/nph.1828':'10.1111/nph.18287',
    '10.1111/nph.1950': '10.1111/nph.19506',
    'doi: 10.1111/j.1364-3703.2012.00840.':'https://doi.org/10.1111/j.1364-3703.2012.00840.x',
    'doi.org/10.1073/pnas.180819411':'DOI: 10.1073/pnas.1808194115',
    'DOI: 10.7554/eLife.10856.001': '10.7554/eLife.10856',
    'https://doi.org/10.1073/pnas.200048211': 'https://doi.org/10.1073/pnas.2000482119',
    '10.1073/pnas.240073712':'https://doi.org/10.1073/pnas.2400737121'
}

def normalize_columns(df):
    # Normalize plant species
    df['normalized_plant_species'] = (
        df['plant_species']
        .map(plant_species_mapping)
        .replace([np.nan, None], 'non-specific')
    )
    
    # Normalize area/category (assuming the column is called 'area')
    df['normalized_area'] = (
        df['area']
        .map(area_mapping)
        .fillna(df['area'])  # Fallback for any unmapped values
    )

    # Correct source
    df['source'] = (
        df['source']
        .map(source_mapping)  # Apply mapping
        .fillna(df['source'])  # Preserve unmapped values
        .replace([np.nan, None], 'non-specific')  # Handle original NaN/None
    )
    
    # Convert string representations of lists to actual lists (if needed)
    def convert_to_list(x):
        if isinstance(x, str) and x.startswith('['):
            return eval(x)
        return x
    
    df['normalized_plant_species'] = df['normalized_plant_species'].apply(convert_to_list)
    
    return df


if __name__ == '__main__':
    df = pd.read_csv('data\mcq_results_all_shuffles_clean.csv')
    df = normalize_columns(df)
    df.to_json('data/normalized.json', orient='records', indent=2)