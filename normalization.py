import pandas as pd
import numpy as np
import re
from collections import defaultdict

# Load the dataset
df = pd.read_csv('ALL_DATA_2.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def clean_text(text):
    """Clean text by removing extra whitespace and special characters"""
    if pd.isna(text):
        return text

    # Convert to string
    text = str(text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)

    # Remove zero-width spaces and other invisible characters
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff]', '', text)

    return text


def normalize_for_matching(text):
    """Normalize text for case-insensitive matching"""
    if pd.isna(text):
        return text
    text = clean_text(text)
    return text.lower().strip()


# ============================================================================
# CLEAN ALL TEXT FIELDS
# ============================================================================

print("=" * 80)
print("CLEANING DATA")
print("=" * 80)

fields_to_clean = ['weather', 'time_of_day', 'season', 'activity', 'mood', 'country']

for field in fields_to_clean:
    if field in df.columns:
        # Show sample before cleaning
        print(f"\n{field} - Sample before cleaning:")
        sample_vals = df[field].dropna().head(3).tolist()
        for val in sample_vals:
            print(f"  '{val}' (repr: {repr(val)})")

        # Clean the field
        df[field] = df[field].apply(clean_text)

        # Show sample after cleaning
        print(f"{field} - Sample after cleaning:")
        sample_vals = df[field].dropna().head(3).tolist()
        for val in sample_vals:
            print(f"  '{val}' (repr: {repr(val)})")

print(f"\n✓ Data cleaning complete\n")

# ============================================================================
# NORMALIZATION MAPPINGS (Using normalized keys for matching)
# ============================================================================

# Weather normalization - target: Sunny, Rainy, Cloudy, Snowy, Not Clear
weather_mapping = {
    'sunny': 'Sunny',
    'clear': 'Sunny',
    'clear / sunny': 'Sunny',
    'clear night': 'Sunny',
    'clear night sky': 'Sunny',
    'cold / clear': 'Sunny',

    'cloudy': 'Cloudy',
    'cloudy sunset': 'Cloudy',
    'partly cloudy': 'Cloudy',
    'partly cloudy / sunny': 'Cloudy',
    'windy': 'Cloudy',

    'rainy': 'Rainy',
    'raniy': 'Rainy',

    'snowy': 'Snowy',
    'cold': 'Snowy',

    'not clear': 'Not Clear',
    'not clear (night lighting)': 'Not Clear',
    'no clear': 'Not Clear'
}

# Time of Day normalization - target: Morning, Afternoon, Evening
time_of_day_mapping = {
    'morning': 'Morning',
    'morning,': 'Morning',
    'sunrise': 'Morning',

    'afternoon': 'Afternoon',
    'afternoonafternoon': 'Afternoon',
    'noon': 'Afternoon',
    'daytime': 'Afternoon',

    'evening': 'Evening',
    'night': 'Evening',
    'evning': 'Evening',
    'sunset': 'Evening',

    'not clear': 'Not Clear'
}

# Season normalization - target: Spring, Summer, Fall, Winter, Not Clear
season_mapping = {
    'spring': 'Spring',
    'springl': 'Spring',

    'summer': 'Summer',

    'fall': 'Fall',
    'autumn': 'Fall',

    'winter': 'Winter',

    'not clear': 'Not Clear',
    'clear': 'Not Clear'
}

# Activity normalization - keeping main categories (all lowercase for matching)
activity_mapping = {
    # Sightseeing variations
    'sightseeing': 'Sightseeing',
    'sightseein': 'Sightseeing',
    'sightseing': 'Sightseeing',
    'siteseeking': 'Sightseeing',
    'city sightseeing': 'Sightseeing',
    'historical sightseeing': 'Sightseeing',
    'nature sightseeing': 'Sightseeing',
    'countryside sightseeing': 'Sightseeing',
    'scenic viewpoint sightseeing': 'Sightseeing',
    'sightseeing / boat ride': 'Sightseeing',
    'sightseeing / city tour': 'Sightseeing',
    'sightseeing / exploring ruins': 'Sightseeing',
    'sightseeing / heritage tourism / photography': 'Sightseeing',
    'sightseeing / nature viewing': 'Sightseeing',
    'sightseeing / photography': 'Sightseeing',
    'sightseeing / river cruising': 'Sightseeing',
    'sightseeing / walking': 'Sightseeing',
    'sightseeing / walking along the wall': 'Sightseeing',
    'sightseeing / exploring a historic landmark': 'Sightseeing',
    'sightseeing / relaxing by the river': 'Sightseeing',
    'sightseeing / tourism with camels at the pyramids': 'Sightseeing',
    'sightseeing / walking along the river': 'Sightseeing',
    'sightseeing / walking or hiking along the great wall': 'Sightseeing',
    'sightseeing / walking through a historic, picturesque neighborhood': 'Sightseeing',
    'sightseeing and relaxing on the grass': 'Sightseeing',
    'sightseeing and shopping': 'Sightseeing',
    'sightseeing/ hiking': 'Sightseeing',
    'sightseeingexcitement': 'Sightseeing',
    'sight seeing': 'Sightseeing',
    'city exploration': 'Sightseeing',
    'city exploring': 'Sightseeing',
    'city stroll': 'Sightseeing',
    'city walk': 'Sightseeing',
    'city strolling': 'Sightseeing',
    'city tour': 'Sightseeing',
    'city walking / sightseeing': 'Sightseeing',
    'urban exploration': 'Sightseeing',
    'urban exploring': 'Sightseeing',
    'urban life': 'Sightseeing',
    'tourism': 'Sightseeing',
    'touring': 'Sightseeing',
    'visiting': 'Sightseeing',
    'visiting landmarks': 'Sightseeing',

    # Hiking
    'hiking': 'Hiking',
    'hiking / exploring': 'Hiking',
    'hiking / exploring ruins': 'Hiking',
    'hiking / sightseeing': 'Hiking',
    'hiking and enjoying nature': 'Hiking',
    'hiking and exploring nature': 'Hiking',
    'hiking in the mountains': 'Hiking',
    'skiing or mountain hiking': 'Hiking',

    # Relaxing
    'relaxing': 'Relaxing',
    'relaxing / boat viewing': 'Relaxing',
    'relaxing / boating': 'Relaxing',
    'relaxing / nature viewing': 'Relaxing',
    'relaxing / swimming / beach vacation': 'Relaxing',
    'relaxing / viewing nature': 'Relaxing',
    'relaxing / nature visiting': 'Relaxing',
    'relaxing / sightseeing': 'Relaxing',
    'exploring / relaxing': 'Relaxing',
    'aurora watching / relaxing': 'Relaxing',
    'meditation': 'Relaxing',
    'resort stay': 'Relaxing',
    'vacation': 'Relaxing',
    'vacationing': 'Relaxing',
    'relaxation': 'Relaxing',
    'relaxed': 'Relaxing',

    # Walking
    'walking': 'Walking',
    'walking / city exploring': 'Walking',
    'walking / commuting / sightseeing': 'Walking',
    'walking / photography': 'Walking',
    'walking / city exploring': 'Walking',
    'walking / village exploring': 'Walking',
    'walking tour': 'Walking',
    'walking around the city': 'Walking',
    'walking in the canal': 'Walking',
    'wandering': 'Walking',
    'strolling': 'Walking',
    'nature walk': 'Walking',

    # Water Sports
    'swimming': 'Water Sports',
    'diving': 'Water Sports',
    'scuba diving': 'Water Sports',
    'snorkeling': 'Water Sports',
    'snorkelling': 'Water Sports',
    'surfing': 'Water Sports',
    'jet skiing': 'Water Sports',
    'sailing': 'Water Sports',
    'boat ride': 'Water Sports',
    'boat riding': 'Water Sports',
    'boat tour': 'Water Sports',
    'boating': 'Water Sports',
    'rowing': 'Water Sports',
    'boat riding / sightseeing': 'Water Sports',
    'boat riding / swimming / sightseeing': 'Water Sports',
    'canoeing': 'Water Sports',
    'canal touring': 'Water Sports',
    'gondola riding': 'Water Sports',

    # Winter Sports
    'skiing': 'Winter Sports',
    'skiing / winter activities': 'Winter Sports',
    'ice skating': 'Winter Sports',
    'skating': 'Winter Sports',
    'snow activities / exploring': 'Winter Sports',

    # Adventure Sports
    'skydiving': 'Adventure Sports',
    'bungee jumping': 'Adventure Sports',
    'hot air ballooning': 'Adventure Sports',
    'hot-air balloon ride': 'Adventure Sports',
    'hot-air balloon riding': 'Adventure Sports',
    'ballooning': 'Adventure Sports',
    'cliff jumping': 'Adventure Sports',

    # Wildlife/Nature
    'safari': 'Wildlife/Nature',
    'wildlife watching': 'Wildlife/Nature',
    'nature viewing': 'Wildlife/Nature',
    'nature watching': 'Wildlife/Nature',
    'nature exploration': 'Wildlife/Nature',
    'mountain viewing': 'Wildlife/Nature',
    'viewing nature / countryside': 'Wildlife/Nature',
    'aurora watching': 'Wildlife/Nature',
    'stargazing': 'Wildlife/Nature',

    # Cultural
    'cultural experience': 'Cultural',
    'cultural exploration': 'Cultural',
    'cultural visit': 'Cultural',
    'cultural tour': 'Cultural',
    'cultural tourism': 'Cultural',
    'history tour': 'Cultural',
    'pilgrimage': 'Cultural',
    'pilgrimage / spiritual visit': 'Cultural',
    'religious pilgrimage': 'Cultural',
    'religious visit': 'Cultural',
    'religious/spiritual visit': 'Cultural',
    'praying': 'Cultural',
    'praying / worshiping': 'Cultural',
    'worship': 'Cultural',
    'worshipping': 'Cultural',
    'breakfast in front of al-aqsa mosque': 'Cultural',

    # Shopping
    'shopping': 'Shopping',
    'traditional shopping': 'Shopping',

    # Other activities
    'biking': 'Cycling',
    'cycling': 'Cycling',
    'camping': 'Camping',
    'photography': 'Photography',
    'exploring': 'Exploring',
    'exploring the desert': 'Exploring',
    'village exploring': 'Exploring',
    'football': 'Sports',
    'sports event': 'Sports',
    'sports / stadium visit': 'Sports',
    'stadium tour': 'Sports',
    'spectating': 'Sports',
    'watching a football match': 'Sports',
    'eating': 'Eating',
    'entertainment': 'Entertainment',
    'gaming and entertainment': 'Entertainment',
    'camel riding': 'Riding',
    'riding a horse': 'Riding',
    'beachgoing': 'Beachgoing',
    'boarding': 'Boarding',
    'celebration': 'Celebration',
    'romance': 'Romance',
    'visiting car dealerships': 'Other',

    # No activity
    'no activity': 'No Activity',
    'none': 'No Activity'
}

# Mood normalization
mood_mapping = {
    'excitement': 'Excitement',
    'exitment': 'Excitement',
    'energy': 'Excitement',
    'vibrant': 'Excitement',
    'amazed': 'Excitement',
    'joy': 'Excitement',
    'magical': 'Excitement',
    'impressed': 'Excitement',

    'happiness': 'Happiness',
    'happiness.': 'Happiness',
    'calm': 'Happiness',
    'calmness': 'Happiness',
    'peace': 'Happiness',
    'peaceful': 'Happiness',
    'peacefulness': 'Happiness',
    'serenity': 'Happiness',
    'tranquility': 'Happiness',

    'curiosity': 'Curiosity',
    'curiosity / awe': 'Curiosity',
    'curiosty': 'Curiosity',
    'curiousity': 'Curiosity',
    'curosity': 'Curiosity',
    'wonder': 'Curiosity',
    'awe': 'Curiosity',
    'awe and adventure': 'Curiosity',
    'spiritual awe': 'Curiosity',

    'nostalgia': 'Nostalgia',
    'historic': 'Nostalgia',
    'respect': 'Nostalgia',
    'reverence': 'Nostalgia',
    'reverence / awe': 'Nostalgia',
    'humility': 'Nostalgia',
    'spirituality': 'Nostalgia',

    'adventure': 'Adventure',
    'adventurous': 'Adventure',
    'advanture': 'Adventure',
    'majestic': 'Adventure',
    'inspiration': 'Adventure',

    'romance': 'Romance',
    'romance.': 'Romance',
    'romantic': 'Romance',
    'romance .': 'Romance',
    'elegant': 'Romance',

    'melancholy': 'Melancholy',
    'melanchol': 'Melancholy'
}

# Country normalization (all lowercase for matching)
country_mapping = {
    'usa': 'USA',
    'united states': 'USA',
    'united states of america': 'USA',
    'california': 'USA',
    'california, usa': 'USA',
    'miami': 'USA',
    'new york': 'USA',
    'north pole, alaska': 'USA',

    'uk': 'UK',
    'united kingdom': 'UK',
    'england': 'UK',
    'scotland': 'UK',
    'united kingdom (scotland)': 'UK',
    'wales': 'UK',
    'london': 'UK',

    'france': 'France',
    'paris': 'France',

    'italy': 'Italy',
    'italy (lake como)': 'Italy',

    'germany': 'Germany',
    'switzerland': 'Switzerland',
    'interlaken': 'Switzerland',

    'netherlands': 'Netherlands',
    'the netherlands': 'Netherlands',
    'holland': 'Netherlands',

    'china': 'China',
    'hong kong': 'China',

    'japan': 'Japan',
    'tokyo/japan': 'Japan',

    'egypt': 'Egypt',
    'eygypt': 'Egypt',

    'uae': 'UAE',
    'united arab emirates': 'UAE',

    'turkey': 'Turkey',
    'turkiye': 'Turkey',
    'türkiye': 'Turkey',

    'south africa': 'South Africa',
    'russia': 'Russia',
    'sweden': 'Sweden',

    'indonesia': 'Indonesia',
    'bali': 'Indonesia',
    'bali indonesia.': 'Indonesia',

    'saudi arabia': 'Saudi Arabia',
    'suadi arabia': 'Saudi Arabia',
    'mecca': 'Saudi Arabia',

    'czech republic': 'Czech Republic',
    'czechia': 'Czech Republic',
    'prague': 'Czech Republic',

    'austria': 'Austria',
    'vienna': 'Austria',

    'greece': 'Greece',
    'santorini': 'Greece',

    'maldives': 'Maldives',
    'maldive': 'Maldives',

    'south korea': 'South Korea',
    'korea': 'South Korea',

    'peru': 'Peru',
    'preu': 'Peru',

    'romania': 'Romania',

    'palestine': 'Palestine',
    'jerusalem': 'Palestine',

    'chile': 'Chile',
    'easter island': 'Chile',
    'rapa nui': 'Chile',

    'antarctica': 'Antarctica',
    'antarctica (continent, not a country)': 'Antarctica',

    'new zealand': 'New Zealand',
    'norway': 'Norway',
    'iceland': 'Iceland',
    'spain': 'Spain',
    'portugal': 'Portugal',
    'morocco': 'Morocco',
    'india': 'India',
    'thailand': 'Thailand',
    'vietnam': 'Vietnam',
    'cambodia': 'Cambodia',
    'nepal': 'Nepal',
    'bhutan': 'Bhutan',
    'afghanistan': 'Afghanistan',
    'pakistan': 'Pakistan',
    'lebanon': 'Lebanon',
    'jordan': 'Jordan',
    'syria': 'Syria',
    'yemen': 'Yemen',
    'oman': 'Oman',
    'qatar': 'Qatar',
    'kazakhstan': 'Kazakhstan',
    'uzbekistan': 'Uzbekistan',
    'georgia': 'Georgia',
    'albania': 'Albania',
    'croatia': 'Croatia',
    'serbia': 'Serbia',
    'slovenia': 'Slovenia',
    'poland': 'Poland',
    'hungary': 'Hungary',
    'denmark': 'Denmark',
    'finland': 'Finland',
    'greenland': 'Greenland',
    'ireland': 'Ireland',
    'malta': 'Malta',
    'cyprus': 'Cyprus',
    'monaco': 'Monaco',
    'andorra': 'Andorra',
    'canada': 'Canada',
    'arctic circle': 'Canada',
    'mexico': 'Mexico',
    'brazil': 'Brazil',
    'argentina': 'Argentina',
    'colombia': 'Colombia',
    'bolivia': 'Bolivia',
    'costa rica': 'Costa Rica',
    'australia': 'Australia',
    'malaysia': 'Malaysia',
    'singapore': 'Singapore',
    'philippines': 'Philippines',
    'myanmar': 'Myanmar',
    'kenya': 'Kenya',
    'tanzania': 'Tanzania',
    'zambia': 'Zambia',
    'zimbabwe': 'Zimbabwe',
    'zimbabwe/zambia': 'Zimbabwe',
    'seychelles': 'Seychelles',
    'bahamas': 'Bahamas',
    'french polynesia': 'French Polynesia',
    'tunisia': 'Tunisia',
    'algeria': 'Algeria',
    'taiwan': 'Taiwan',
    'north korea': 'North Korea',
    'ukraine': 'Ukraine',
    'china and mongolia': 'Mongolia'
}


# ============================================================================
# NORMALIZATION FUNCTION
# ============================================================================

def normalize_field(df, field_name, mapping_dict):
    """Normalize a field using case-insensitive matching"""

    normalized_values = []
    unmapped_values = set()

    for value in df[field_name]:
        if pd.isna(value):
            normalized_values.append(value)
            continue

        # Normalize for matching (lowercase, stripped)
        normalized_key = normalize_for_matching(value)

        # Look up in mapping
        if normalized_key in mapping_dict:
            normalized_values.append(mapping_dict[normalized_key])
        else:
            # Not found in mapping
            normalized_values.append(value)  # Keep original
            unmapped_values.add(value)

    return pd.Series(normalized_values), unmapped_values


# ============================================================================
# APPLY NORMALIZATION
# ============================================================================

print("=" * 80)
print("NORMALIZING FIELDS")
print("=" * 80)

# Create a copy for normalized data
df_normalized = df.copy()

# Normalize each field
fields_to_normalize = {
    'weather': weather_mapping,
    'time_of_day': time_of_day_mapping,
    'season': season_mapping,
    'activity': activity_mapping,
    'mood': mood_mapping,
    'country': country_mapping
}

unmapped_report = {}

for field, mapping in fields_to_normalize.items():
    print(f"\nNormalizing '{field}'...")
    normalized_col, unmapped = normalize_field(df, field, mapping)
    df_normalized[field] = normalized_col

    # Store unmapped values
    if len(unmapped) > 0:
        unmapped_report[field] = sorted(unmapped)
        print(f"  ⚠ Warning: {len(unmapped)} unmapped values found")
        # Show first few unmapped values
        for val in list(unmapped)[:5]:
            print(f"     - '{val}' (repr: {repr(val)})")
    else:
        print(f"  ✓ All values mapped successfully")

# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

print("\n" + "=" * 80)
print("NORMALIZATION SUMMARY")
print("=" * 80)

for field in fields_to_normalize.keys():
    print(f"\n{field.upper()}:")
    print(f"  Original unique values: {df[field].nunique()}")
    print(f"  Normalized unique values: {df_normalized[field].nunique()}")
    unique_vals = sorted([v for v in df_normalized[field].dropna().unique() if v])
    print(f"  Normalized categories: {unique_vals}")

# Report unmapped values
if unmapped_report:
    print("\n" + "=" * 80)
    print("UNMAPPED VALUES REPORT")
    print("=" * 80)
    for field, values in unmapped_report.items():
        print(f"\n{field}: ({len(values)} unmapped)")
        for val in values:
            print(f"  - '{val}' | repr: {repr(val)}")

# ============================================================================
# SAVE NORMALIZED DATA
# ============================================================================

output_file = 'ALL_DATA_2_NORMALIZED.csv'
df_normalized.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n" + "=" * 80)
print(f"✓ Normalized data saved to: {output_file}")
print(f"  Total rows: {len(df_normalized)}")
print(f"  Total columns: {len(df_normalized.columns)}")
print("=" * 80)

# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("VALUE DISTRIBUTION (Top 10 per field)")
print("=" * 80)

for field in fields_to_normalize.keys():
    print(f"\n{field.upper()}:")
    value_counts = df_normalized[field].value_counts()
    for val, count in value_counts.head(10).items():
        percentage = (count / len(df_normalized)) * 100
        print(f"  {val}: {count} ({percentage:.1f}%)")

print("\n" + "=" * 80)
print("NORMALIZATION COMPLETE!")
print("=" * 80)