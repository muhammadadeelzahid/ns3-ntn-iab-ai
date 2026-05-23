#include "dash-metadata.h"

#include "ns3/log.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("DashMetadata");

namespace
{
std::string
Trim(const std::string& value)
{
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
    {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
    {
        --end;
    }
    return value.substr(start, end - start);
}

std::vector<std::string>
SplitCsvLine(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ','))
    {
        out.push_back(Trim(cell));
    }
    if (!line.empty() && line.back() == ',')
    {
        out.push_back("");
    }
    return out;
}

} // namespace

DashMetadataIndex::DashMetadataIndex()
    : m_loaded(false)
{
}

bool
DashMetadataIndex::Load(const std::string& csvPath, std::string* errorMessage)
{
    m_segmentSizeByRate.clear();
    m_loaded = false;

    std::ifstream in(csvPath.c_str());
    if (!in.is_open())
    {
        if (errorMessage)
        {
            *errorMessage = "failed to open CSV: " + csvPath;
        }
        return false;
    }

    std::string headerLine;
    if (!std::getline(in, headerLine))
    {
        if (errorMessage)
        {
            *errorMessage = "empty CSV: " + csvPath;
        }
        return false;
    }

    const auto headers = SplitCsvLine(headerLine);
    std::unordered_map<std::string, size_t> colIndex;
    for (size_t i = 0; i < headers.size(); ++i)
    {
        colIndex[headers[i]] = i;
    }

    const std::vector<std::pair<uint32_t, std::string>> rateColumns = {
        {250000, "size_bytes_250k"},
        {750000, "size_bytes_750k"},
        {1500000, "size_bytes_1500k"},
        {3000000, "size_bytes_3000k"},
        {6000000, "size_bytes_6000k"},
        {12000000, "size_bytes_12000k"},
    };

    auto segIdxIt = colIndex.find("segment_index");
    if (segIdxIt == colIndex.end())
    {
        if (errorMessage)
        {
            *errorMessage = "missing required column: segment_index";
        }
        return false;
    }
    for (const auto& rateCol : rateColumns)
    {
        if (colIndex.find(rateCol.second) == colIndex.end())
        {
            if (errorMessage)
            {
                *errorMessage = "missing required column: " + rateCol.second;
            }
            return false;
        }
    }

    std::string line;
    uint32_t rowsParsed = 0;
    while (std::getline(in, line))
    {
        if (Trim(line).empty())
        {
            continue;
        }

        const auto cells = SplitCsvLine(line);
        const size_t segIdxCol = segIdxIt->second;
        if (segIdxCol >= cells.size())
        {
            continue;
        }

        uint32_t segmentIndex = 0;
        try
        {
            segmentIndex = static_cast<uint32_t>(std::stoul(cells[segIdxCol]));
        }
        catch (const std::exception&)
        {
            continue;
        }

        if (segmentIndex >= m_segmentSizeByRate.size())
        {
            m_segmentSizeByRate.resize(segmentIndex + 1);
        }

        auto& row = m_segmentSizeByRate[segmentIndex];
        bool hasAnyRate = false;
        for (const auto& rateCol : rateColumns)
        {
            const auto colIt = colIndex.find(rateCol.second);
            if (colIt == colIndex.end() || colIt->second >= cells.size())
            {
                continue;
            }
            try
            {
                const double sizeVal = std::stod(cells[colIt->second]);
                if (sizeVal > 0.0)
                {
                    row[rateCol.first] = static_cast<uint32_t>(std::max(1.0, std::round(sizeVal)));
                    hasAnyRate = true;
                }
            }
            catch (const std::exception&)
            {
                continue;
            }
        }

        if (hasAnyRate)
        {
            ++rowsParsed;
        }
    }

    if (rowsParsed == 0)
    {
        if (errorMessage)
        {
            *errorMessage = "no valid segment rows parsed from CSV";
        }
        return false;
    }

    m_loaded = true;
    return true;
}

bool
DashMetadataIndex::IsLoaded() const
{
    return m_loaded;
}

uint32_t
DashMetadataIndex::GetSegmentCount() const
{
    return static_cast<uint32_t>(m_segmentSizeByRate.size());
}

bool
DashMetadataIndex::GetSegmentSizeBytes(uint32_t segmentId, uint32_t bitrateBps, uint32_t& sizeBytes) const
{
    if (!m_loaded || segmentId >= m_segmentSizeByRate.size())
    {
        return false;
    }
    const auto& row = m_segmentSizeByRate[segmentId];
    const auto it = row.find(bitrateBps);
    if (it == row.end())
    {
        return false;
    }
    sizeBytes = it->second;
    return true;
}

} // namespace ns3
